"""Failure, identity, artifact and native-stop tests for unattended collection."""
import copy
import dataclasses
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

import test_canvas_debugger as fixtures
from exp_code.positional_intervention import run_batch as batch
from exp_code.positional_intervention.analyze import validate_probe, sha256
from exp_code.positional_intervention.batch_gates import verify_mbpp
from exp_code.positional_intervention.resident_position import probe


class Tokenizer:
    def decode(self, ids):
        return str(np.asarray(ids).tolist())


class BatchLifecycleTests(unittest.TestCase):
    def test_gate_exception_is_nonzero_without_input_and_preserves_attempt(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(batch, 'source_files', return_value={}), \
                mock.patch.object(batch, 'execute', side_effect=AssertionError('native gate failed')), \
                mock.patch('builtins.input', side_effect=AssertionError('must never prompt')):
            args = ['--output', directory]
            self.assertEqual(batch.main(args), 1)
            manifest = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertTrue((Path(directory)/manifest['attempts'][0]['path']/'failure.json').is_file())
            self.assertTrue(all(i['status']=='pending' for i in manifest['items'].values()))
            self.assertEqual(batch.main(args+['--resume']), 1)
            revised = json.loads((Path(directory)/'manifest.json').read_text())
            self.assertEqual(len(revised['attempts']), 2)
            self.assertNotEqual(*[a['path'] for a in revised['attempts']])

    def test_resume_rejects_changed_uncommitted_source_before_execution(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(batch, 'source_files', return_value={}), \
                mock.patch.object(batch, 'execute', side_effect=batch.GPUUnavailable('no allocation')):
            self.assertEqual(batch.main(['--output', directory]), 2)
            before = (Path(directory)/'manifest.json').read_bytes()
            with mock.patch.object(batch, 'source_files', return_value={'uncommitted.py': 'new hash'}), \
                    mock.patch.object(batch, 'execute') as execute:
                self.assertEqual(batch.main(['--output', directory, '--resume']), 1)
                execute.assert_not_called()
            self.assertEqual((Path(directory)/'manifest.json').read_bytes(), before)

    def test_resume_directory_is_not_proof_of_completion(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(batch, 'source_files', return_value={}):
            self.assertEqual(batch.main(['--output', directory, '--resume']), 1)
            self.assertFalse((Path(directory)/'manifest.json').exists())

    def test_scheduler_end_time_and_required_allocation(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            with self.assertRaises(batch.GPUUnavailable):
                batch.scheduler()
        end = batch.datetime.fromtimestamp(batch.time.time()+300).isoformat(timespec='seconds')
        with mock.patch.dict('os.environ', {'SLURM_JOB_ID': '123'}), \
                mock.patch.object(batch, 'command', return_value=f'JobId=123 JobState=RUNNING EndTime={end}'):
            self.assertGreater(batch.scheduler()['remaining_seconds'], 295)
            self.assertLess(batch.scheduler()['remaining_seconds'], 301)

    def test_checkpoint_declared_hash_is_not_accepted_without_content_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer.model').write_bytes(b'correct')
            entries = [dict(name='tokenizers/tokenizer_gemma4.model', size=7, sha256=sha256(root/'tokenizer.model'))]
            (root/'verified-manifest.json').write_text(json.dumps(dict(files=entries, sha256=batch.digest(entries))))
            self.assertEqual(batch.checkpoint_identity(root)['tokenizer_sha256'], entries[0]['sha256'])
            (root/'tokenizer.model').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload hash mismatch'):
                batch.checkpoint_identity(root)

    def test_design_scope_and_original_batch_are_enforced(self):
        design = batch.read_json(batch.ROOT/'configs/direction2_pilot.json')
        previous = batch.read_json(batch.ROOT/'configs/debugger_run.json')
        _, focus, seed = batch.validate_design(design, previous)
        self.assertEqual((focus, seed), (2, 100008))
        for field, value in [('horizon', 8), ('source', 41), ('case_id', 'mbpp:59')]:
            changed = copy.deepcopy(design)
            changed[field] = value
            with self.assertRaises(ValueError):
                batch.validate_design(changed, previous)

    def test_eligibility_is_baseline_only_and_records_failures(self):
        design = batch.read_json(batch.ROOT/'configs/direction2_pilot.json')
        logits = np.full((1, 64, 12235), -np.inf, dtype=np.float32)
        logits[:, :, 0] = 0.
        for pos, p in ((42, .02), (43, .03)):
            logits[0, pos, 12234] = np.log(p)
            logits[0, pos, 0] = np.log(1-p)
        engine = SimpleNamespace(state=SimpleNamespace(step=2, done=[False],
            sc_logits=logits, canvas=np.zeros((1,64), dtype=int)), _editable=lambda *args: None)
        item = design['items'][0]
        self.assertTrue(batch.eligibility(engine, design, item, 0)['eligible'])
        logits[0, 42, 12234] = -np.inf
        result = batch.eligibility(engine, design, item, 0)
        self.assertFalse(result['eligible'])
        self.assertIn('nonfinite anchor logits', result['reasons'])
        self.assertIn('source_min threshold not met', result['reasons'])
        engine.state.done = [True]
        engine._editable = mock.Mock(side_effect=ValueError('Sample already stopped'))
        result = batch.eligibility(engine, design, item, 0)
        self.assertFalse(result['active'])
        self.assertFalse(result['eligible'])
        self.assertIn('Sample already stopped', result['reasons'])

    def test_direct_native_verifier_failure_blocks_mbpp_and_collection(self):
        from exp_code.semantic_intervention import debug_console
        from exp_code.positional_intervention import batch_gates
        design = batch.read_json(batch.ROOT/'configs/direction2_pilot.json')
        previous = batch.read_json(batch.ROOT/'configs/debugger_run.json')
        tokenizer = SimpleNamespace(FORMAT=None, decode=lambda ids: str(ids))
        chat = SimpleNamespace(canvas_length=256, max_denoising_steps=48, max_out_length=1024,
            cache_length=4096, early_stop_fn='native', logit_shaper='native', sampling=None,
            sc_mode='full', sample_from_predictions='native', _get_inputs=lambda **kwargs: np.zeros((4,2)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/'attempt').mkdir()
            attempt = dict(path='attempt', id='test', progress=[])
            manifest = dict(base_identity={}, items={i['id']: dict(status='pending', design=i) for i in design['items']})
            with mock.patch.object(batch, 'preflight_gpu', return_value={'nvidia_smi': 'gpu, uuid, driver, 96000 MiB'}), \
                    mock.patch.object(batch, 'scheduler', return_value={'remaining_seconds': 10000}), \
                    mock.patch.object(batch, 'checkpoint_identity', return_value={}), \
                    mock.patch('exp_code.baselines.model_loading.load_model', return_value=(object(), object(), tokenizer, None)) as load, \
                    mock.patch('exp_code.common.sc_sampler.SCSampler', return_value=chat), \
                    mock.patch('dialog.Conversation') as conversation, \
                    mock.patch.object(debug_console, 'verify_checkpoint', side_effect=AssertionError('real verifier failure')) as native, \
                    mock.patch.object(batch_gates, 'verify_mbpp') as mbpp, \
                    mock.patch('builtins.input', side_effect=AssertionError('must not prompt')):
                conversation.return_value.as_text.return_value = 'rendered test prompt'
                with self.assertRaisesRegex(AssertionError, 'real verifier failure'):
                    batch.execute(design, previous, output, manifest, attempt)
                load.assert_called_once()
                native.assert_called_once()
                mbpp.assert_not_called()
            self.assertTrue(all(v['status']=='pending' for v in manifest['items'].values()))


class TinyBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DebuggerTests.setUpClass()

    def engine(self, sampler=None):
        return fixtures.DebuggerTests().engine(sampler)

    def make_probe(self, engine, output, horizon=5):
        return probe(engine, source=0, target=1, token=0, fraction=.5,
                     horizon=horizon, batch=0, case_id='tiny:1', seed=11,
                     tokenizer=Tokenizer(), output=output, neighbors=[1])

    def test_cap_horizon_and_resume_require_all_artifacts(self):
        engine = self.engine()
        engine.until(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'probe'
            report = self.make_probe(engine, folder)
            self.assertEqual(report['branches']['baseline']['observed_followups'], 3)
            self.assertEqual(report['branches']['baseline']['stop_reason'], 'step_cap')
            validate_probe(folder)
            item = dict(status='complete', identity_sha256='identity', probe_path='probe',
                artifacts={str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*') if p.is_file()},
                probe_request={'case_id': 'tiny:1', 'base_completed_step': 1},
                native_gate_path='native.json', mbpp_gate_path='mbpp.json')
            (root/'native.json').write_text(json.dumps({'passed': True}))
            (root/'mbpp.json').write_text(json.dumps({'status': 'passed', 'checks': {'parity': True}}))
            item['gate_artifacts'] = {p: sha256(root/p) for p in ('native.json', 'mbpp.json')}
            self.assertTrue(batch.completed_valid(root, item, 'identity'))
            self.assertFalse(batch.completed_valid(root, item, 'identity', {'base_completed_step': 2}))
            self.assertFalse(batch.completed_valid(root, item, 'different source'))
            self.assertFalse(batch.completed_valid(root, dict(item, status='running'), 'identity'))
            (folder/'reverse/position_trace.json').unlink()
            self.assertFalse(batch.completed_valid(root, item, 'identity'))

    def test_adaptive_stop_observes_one_step_and_censors_four(self):
        @dataclasses.dataclass(frozen=True)
        class StopAtTwo:
            def should_stop(self, *, step, canvas, **kwargs):
                return jnp.full((canvas.shape[0],), step >= 1)
        loop = dataclasses.replace(fixtures.DebuggerTests.loop, early_stop_fn=StopAtTwo())
        engine = self.engine(loop)
        engine.until(1)
        with tempfile.TemporaryDirectory() as directory:
            report = self.make_probe(engine, Path(directory)/'probe')
            for branch in report['branches'].values():
                self.assertEqual(branch['observed_followups'], 1)
                self.assertEqual(branch['completed_step'], 2)
                self.assertEqual(branch['stop_reason'], 'adaptive')
                self.assertTrue(branch['focus_stopped'])
            validate_probe(Path(directory)/'probe')

    def test_mid_probe_failure_restores_trace_and_keeps_incomplete_attempt(self):
        engine = self.engine()
        engine.until(1)
        original = engine.state
        original_step = engine.step
        calls = 0
        def fail_later():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('interrupted second branch')
            return original_step()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(engine, 'step', side_effect=fail_later):
            folder = Path(directory)/'probe'
            with self.assertRaisesRegex(RuntimeError, 'interrupted second branch'):
                self.make_probe(engine, folder, horizon=2)
            report = batch.read_json(folder/'position_report.json')
            self.assertEqual(report['status'], 'failed_do_not_interpret')
            self.assertTrue(report['restored_base_state'])
            self.assertTrue(report['restored_base_trace'])
            self.assertTrue((folder/'baseline/branch.json').exists())
            with self.assertRaises(ValueError):
                validate_probe(folder)
        self.assertIs(engine.state, original)
        self.assertEqual(len(engine.recorder.events), 1)
        self.assertEqual(engine.edits, [])

    def test_restore_failure_invalidates_a_previously_passed_probe(self):
        engine = self.engine()
        engine.until(1)
        restore = engine.restore
        calls = 0
        def fail_final_restore(name):
            nonlocal calls
            calls += 1
            if calls == 6:
                raise RuntimeError('restore failed')
            return restore(name)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(engine, 'restore', side_effect=fail_final_restore):
            folder = Path(directory)/'probe'
            with self.assertRaisesRegex(RuntimeError, 'restore failed'):
                self.make_probe(engine, folder, horizon=1)
            report = batch.read_json(folder/'position_report.json')
            self.assertEqual(report['status'], 'failed_do_not_interpret')
            self.assertIn('restoration_error', report)

    def test_fresh_prefill_full_state_and_matched_native_gate_on_tiny_model(self):
        loop = fixtures.DebuggerTests.loop
        params = fixtures.DebuggerTests.params
        @dataclasses.dataclass
        class TinyChat:
            canvas_length: int = 3
            max_out_length: int = 12
            max_denoising_steps: int = 4
            sampling: object = None
            sample_from_predictions: object = None
            def _initialize_sampler_loop(self, sampling):
                return dataclasses.replace(loop, sample_from_predictions=self.sample_from_predictions) if self.sample_from_predictions else loop
            def sample(self, prompts, *, rng, max_new_tokens, return_state):
                if max_new_tokens != 3:
                    raise AssertionError('Native length is unmatched')
                sampler = self._initialize_sampler_loop(None)
                tokens = sampler.sample_next_canvas(canvas_length=3, max_denoising_steps=4,
                    batch_size=1, cache=None, params=params, rng=jax.random.PRNGKey(11))
                return SimpleNamespace(tokens=tokens)
        def prepare(*args):
            return self.engine(), SimpleNamespace(rng=jax.random.PRNGKey(11), done=jnp.array([False]))
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('exp_code.positional_intervention.batch_gates.prepare_canvas', side_effect=prepare):
            report = verify_mbpp(TinyChat(), ['tiny'], 11, 0, Path(directory)/'gate')
            self.assertEqual(report['status'], 'passed')
            self.assertTrue(all(report['checks'].values()))
        calls = 0
        def inconsistent(*args):
            nonlocal calls
            engine, initial = prepare()
            calls += 1
            if calls == 2:
                engine.state = dataclasses.replace(engine.state, rng=jax.random.PRNGKey(99))
            return engine, initial
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch('exp_code.positional_intervention.batch_gates.prepare_canvas', side_effect=inconsistent):
            with self.assertRaisesRegex(AssertionError, 'prefills differ'):
                verify_mbpp(TinyChat(), ['tiny'], 11, 0, Path(directory)/'gate')
            self.assertEqual(batch.read_json(Path(directory)/'gate/gate.json')['status'], 'failed')
            self.assertTrue((Path(directory)/'gate/prefill_2_full_identity.json').exists())


if __name__ == '__main__':
    unittest.main()
