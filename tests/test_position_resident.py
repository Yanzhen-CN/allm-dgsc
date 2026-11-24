"""Tiny-model integration checks; run in the existing allm-direction2 environment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import test_canvas_debugger as fixtures
from exp_code.positional_intervention.resident_position import move, probe, _isolation


class PositionResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.DebuggerTests.setUpClass()

    def engine(self):
        return fixtures.DebuggerTests().engine()

    def test_edit_isolation_noop_and_restore(self):
        engine = self.engine()
        engine.until(1)
        engine.save('base')
        base = engine.state
        move(engine, 0, 1, 0, 0., 0)
        self.assertIs(engine.state, base)
        move(engine, 0, 1, 0, .5, 0)
        _isolation(base, engine.state, 0, 0, 1, engine.cache, engine)
        self.assertGreater(float(np.linalg.norm(np.asarray(engine.state.sc_embeddings, dtype=float)
            - np.asarray(base.sc_embeddings, dtype=float))), 0.)
        engine.restore('base')
        self.assertIs(engine.state, base)

    def test_probe_restores_base_and_exports_validated_branches(self):
        engine = self.engine()
        engine.until(1)
        base = engine.state
        engine.save('user_base')
        class Tokenizer:
            def decode(self, ids):
                return str(np.asarray(ids).tolist())
        with tempfile.TemporaryDirectory() as directory:
            report = probe(engine, source=0, target=1, token=0, fraction=.5,
                horizon=2, batch=0, case_id='tiny:1', seed=11, tokenizer=Tokenizer(),
                output=Path(directory)/'probe', neighbors=[1])
            self.assertEqual(report['status'], 'passed')
            self.assertTrue(all(report['checks'].values()))
            self.assertEqual(len(report['branches']), 5)
            self.assertTrue((Path(directory)/'probe/forward/position_probabilities.csv').exists())
        self.assertIs(engine.state, base)
        self.assertEqual(list(engine.snapshots), ['user_base'])

    def test_failed_probe_restores_state_and_marks_results_invalid(self):
        from unittest.mock import patch
        import json
        engine = self.engine()
        engine.until(1)
        base = engine.state
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'probe'
            with patch.object(engine, 'step', side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError, 'test failure'):
                    probe(engine, source=0, target=1, token=0, fraction=.5,
                        horizon=1, batch=0, case_id='tiny:1', seed=11,
                        tokenizer=None, output=output)
            report = json.loads((output/'position_report.json').read_text())
            self.assertEqual(report['status'], 'failed_do_not_interpret')
        self.assertIs(engine.state, base)
        self.assertEqual(engine.snapshots, {})


if __name__ == '__main__':
    unittest.main()
