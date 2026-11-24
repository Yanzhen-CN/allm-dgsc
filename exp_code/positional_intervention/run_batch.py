"""Frozen MBPP277 pilot. Load weights once; fail closed; never prompt for input."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback

from exp_code.common.debug_config import resolve_batch
from exp_code.baselines.data import digest
from .analyze import read_json, write_json, sha256, validate_probe, BRANCHES

ROOT = Path(__file__).resolve().parents[2]
ENV_KEYS = ('JAX_PLATFORMS', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'XLA_PYTHON_CLIENT_MEM_FRACTION',
            'XLA_FLAGS', 'JAX_ENABLE_X64', 'JAX_DEFAULT_MATMUL_PRECISION', 'JAX_COMPILATION_CACHE_DIR',
            'JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', 'TF_ENABLE_ONEDNN_OPTS',
            'NVIDIA_TF32_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONPATH', 'LD_LIBRARY_PATH')


class GPUUnavailable(RuntimeError):
    pass


class TimeLimit(RuntimeError):
    pass


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()


def source_files():
    return {str(p.relative_to(ROOT)): sha256(p) for directory in ('exp_code', 'dg/gemma', 'tests')
            for p in sorted((ROOT/directory).rglob('*.py'))}


def environment():
    versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}
    origins = {name: importlib.util.find_spec(name).origin for name in ('gemma', 'jax', 'flax', 'dialog')}
    if Path(origins['gemma']).resolve() != ROOT/'dg/gemma/__init__.py':
        raise ValueError('Imported Gemma source is outside the hashed repository')
    return dict(python=sys.version, executable=str(Path(sys.executable).resolve()), module_origins=origins,
        dependencies=dict(sorted(versions.items())), flags={key: os.environ.get(key) for key in ENV_KEYS},
        hostname=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID'),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def validate_design(design, previous):
    expected = {(2, 12234, .5), (2, 2060, .5), (4, 12234, .5), (4, 2060, .5),
                (2, 12234, 1.), (2, 2060, 1.)}
    if len(design['items']) != 6 or {(i['step'], i['token'], i['fraction']) for i in design['items']} != expected:
        raise ValueError('This entrypoint permits only the frozen six-item pilot')
    ids = [item['id'] for item in design['items']]
    if len(set(ids)) != 6 or any(not i or Path(i).name != i or i in ('.', '..') for i in ids):
        raise ValueError('Invalid item IDs')
    if (design['case_id'], design['source'], design['target'], design['horizon'],
            design['canvas_length'], design['max_denoising_steps']) != ('mbpp:277', 42, 43, 5, 256, 48):
        raise ValueError('Pilot scope changed')
    if design['eligibility'] != {'12234': {'source_min': .01, 'target_min': .01},
                                  '2060': {'source_min': .2, 'target_min': 0.}}:
        raise ValueError('Development thresholds changed')
    batch, focus, seed = resolve_batch(previous, design['case_id'])
    if ([c['id'] for c in batch], focus, seed) != (design['expected_batch'], design['expected_focus'], design['expected_seed']):
        raise ValueError('Original batch/focus/seed mismatch')
    if (focus, seed, [c['id'] for c in batch]) != (2, 100008, ['mbpp:133', 'mbpp:464', 'mbpp:277', 'mbpp:59']):
        raise ValueError('Pilot is restricted to original MBPP277 batch')
    return batch, focus, seed


def scheduler():
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        raise GPUUnavailable('No SLURM_JOB_ID in this shell; enter an existing GPU allocation. No job submitted.')
    info = command(['scontrol', 'show', 'job', '-o', job])
    fields = dict(part.split('=', 1) for part in shlex.split(info) if '=' in part)
    if fields.get('JobState') != 'RUNNING':
        raise GPUUnavailable('Allocation is not RUNNING')
    try:
        end = datetime.fromisoformat(fields['EndTime']).timestamp()
    except (KeyError, ValueError) as error:
        raise GPUUnavailable('Cannot establish actual scheduler EndTime') from error
    return dict(job_id=job, end_epoch=end, remaining_seconds=end-time.time(), raw=info)


def preflight_gpu():
    allocation = scheduler()
    required = {'JAX_PLATFORMS': 'cuda,cpu', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
                'XLA_PYTHON_CLIENT_MEM_FRACTION': '.95'}
    if any(os.environ.get(k) != v for k, v in required.items()):
        raise GPUUnavailable('Required environment: ' + ' '.join(f'{k}={v}' for k, v in required.items()))
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    selector = ['--id=' + visible] if visible else []
    try:
        apps = command(['nvidia-smi', *selector, '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader'])
        if apps:
            raise GPUUnavailable('Allocated GPU already has compute processes; identify/release them before resuming: ' + apps)
        gpu = command(['nvidia-smi', *selector, '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'])
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        raise GPUUnavailable('Cannot inspect allocated GPU occupancy') from error
    import jax
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise GPUUnavailable('Expected one default GPU; got ' + repr(jax.devices()))
    cpu = jax.devices('cpu')
    if not cpu:
        raise GPUUnavailable('CPU callback backend is required')
    return dict(allocation=allocation, nvidia_smi=gpu, preexisting_compute_processes=apps,
                default_backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                device_kinds=[d.device_kind for d in jax.devices()], cpu_devices=[str(d) for d in cpu],
                effective_jax_config={key: jax.config.values.get(key) for key in (
                    'jax_enable_x64', 'jax_default_matmul_precision', 'jax_default_prng_impl',
                    'jax_threefry_partitionable', 'jax_use_shardy_partitioner',
                    'jax_compilation_cache_dir', 'jax_persistent_cache_min_compile_time_secs')})


def checkpoint_identity(path, progress=lambda *args: None):
    """Verify every manifest payload byte now; a declared digest is insufficient."""
    path = Path(path).resolve()
    manifest = read_json(path/'verified-manifest.json')
    if digest(manifest['files']) != manifest['sha256']:
        raise ValueError('Invalid checkpoint manifest digest')
    prefix = 'checkpoints/diffusiongemma-26B-A4B-it/'
    verified = {}
    for i, item in enumerate(manifest['files']):
        relative = item['name'][len(prefix):] if item['name'].startswith(prefix) else 'tokenizer.model'
        file = path/relative
        if not file.resolve().is_relative_to(path) or relative in verified:
            raise ValueError('Invalid or duplicate manifest entry')
        progress('checkpoint_hash', dict(index=i+1, total=len(manifest['files']), file=relative))
        if file.stat().st_size != item['size'] or sha256(file) != item['sha256']:
            raise ValueError('Checkpoint payload hash mismatch: ' + relative)
        verified[relative] = dict(sha256=item['sha256'], size=item['size'])
    # Also hash local metadata not enumerated by the downloader.
    extra = {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*'))
             if p.is_file() and str(p.relative_to(path)) not in verified}
    return dict(path=str(path), verification='all manifest files content-hashed in this process',
        declared_manifest_sha256=manifest['sha256'], verified_files=verified, extra_files=extra,
        tokenizer_sha256=sha256(path/'tokenizer.model'), manifest_file_sha256=sha256(path/'verified-manifest.json'))


def compatible_identity(base, hardware, checkpoint):
    # Scheduler, host, UUID and PID are provenance, not computation settings.
    gpu_rows = [line.split(',') for line in hardware['nvidia_smi'].splitlines()]
    return dict(base=base, gpu=[dict(name=r[0].strip(), driver=r[2].strip(), memory=r[3].strip()) for r in gpu_rows],
                effective_jax_config=hardware.get('effective_jax_config'), checkpoint=checkpoint)


def completed_valid(output, item_record, identity_sha, expected_request=None):
    if item_record.get('status') != 'complete' or item_record.get('identity_sha256') != identity_sha:
        return False
    folder = output/item_record['probe_path']
    try:
        expected = item_record['artifacts']
        actual = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        if not expected or actual != expected:
            return False
        report, *_ = validate_probe(folder)
        request = expected_request if expected_request is not None else item_record['probe_request']
        if any(report.get(k) != v for k, v in request.items()):
            return False
        gates = item_record['gate_artifacts']
        if not gates or any(not (output/p).is_file() or sha256(output/p) != value for p, value in gates.items()):
            return False
        native = read_json(output/item_record['native_gate_path'])
        mbpp = read_json(output/item_record['mbpp_gate_path'])
        if native.get('passed') is not True or mbpp.get('status') != 'passed' or not mbpp.get('checks') or not all(mbpp['checks'].values()):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def probe_request(design, item):
    return dict(case_id=design['case_id'], seed=design['expected_seed'], focus=design['expected_focus'],
        source_position=design['source'], target_position=design['target'], base_completed_step=item['step'],
        anchor_token_id=item['token'], fraction=item['fraction'], horizon=design['horizon'])


def eligibility(engine, design, item, focus):
    import numpy as np
    from .position_math import probabilities
    state = engine.state
    result = dict(completed_step=state.step, requested_step=item['step'], active=not bool(state.done[focus]),
                  thresholds=design['eligibility'][str(item['token'])], eligible=False, reasons=[])
    if state.step != item['step']:
        result['reasons'].append('batch stopped before requested breakpoint')
    for pos in (design['source'], design['target']):
        try:
            engine._editable(focus, pos)
        except ValueError as error:
            result['reasons'].append(str(error))
    if state.sc_logits is not None:
        rows = np.asarray(state.sc_logits[focus, np.array([design['source'], design['target']])], dtype=np.float32)
        result['finite_anchor_logits'] = bool(np.isfinite(rows[:, item['token']]).all())
        if not result['finite_anchor_logits']:
            result['reasons'].append('nonfinite anchor logits')
        p = probabilities(rows)[:, item['token']]
        result.update(anchor_probabilities=p.tolist(), canvas_token_ids=[int(state.canvas[focus, pos]) for pos in (design['source'], design['target'])])
        result['anchor_present_on_canvas'] = item['token'] in result['canvas_token_ids']
        for i, key in enumerate(('source_min', 'target_min')):
            if p[i] < result['thresholds'][key]:
                result['reasons'].append(f'{key} threshold not met')
    else:
        result['reasons'].append('no prediction logits')
    result['eligible'] = not result['reasons'] and result['active']
    return result


@contextmanager
def process_lock(output):
    """Kernel advisory lock releases on crashes; stale files never imply success."""
    import fcntl
    with (output/'.runner.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def execute(design, previous, output, manifest, attempt):
    def persist():
        write_json(output/'manifest.json', manifest)
    def progress(operation, detail=None):
        event = dict(utc=datetime.now(timezone.utc).isoformat(), operation=operation, detail=detail)
        attempt['progress'].append(event)
        persist()
        print('DIRECTION2', operation, json.dumps(detail), flush=True)
    def guard(operation, estimate):
        allocation = scheduler()
        progress('walltime_check', dict(operation=operation, estimate_seconds=estimate,
            save_margin_seconds=design['walltime']['save_margin_seconds'], **allocation))
        if allocation['remaining_seconds'] < estimate + design['walltime']['save_margin_seconds']:
            raise TimeLimit('Insufficient remaining scheduler time for ' + operation)
    attempt_dir = output/attempt['path']
    from .memory_audit import MemoryAudit, release_engine
    options = manifest.get('execution_options', {})
    cleanup = options.get('memory_policy') == 'cleanup'
    memory = MemoryAudit(attempt_dir/'memory.jsonl', enabled=bool(options.get('memory_policy')))
    attempt['hardware'] = preflight_gpu()
    if options.get('memory_policy'):
        from .check_loading import inspect_checkpoint, require_capacity
        import jax
        capacity = inspect_checkpoint(Path(previous['checkpoint']), attempt_dir)
        physical = float(attempt['hardware']['nvidia_smi'].splitlines()[0].split(',')[3].split()[0])*2**20
        require_capacity(capacity, int(physical), jax.devices()[0].memory_stats().get('bytes_limit'))
        memory('capacity_check_passed')
    guard('checkpoint hashing, weight load and gates', design['walltime']['startup_estimate_seconds'])
    checkpoint = checkpoint_identity(previous['checkpoint'], progress)
    identity = compatible_identity(manifest['base_identity'], attempt['hardware'], checkpoint)
    identity_sha = digest(identity)
    attempt['identity_sha256'] = identity_sha
    write_json(attempt_dir/'identity.json', identity)
    previous_identity = manifest.get('collection_identity_sha256')
    if previous_identity is not None and previous_identity != identity_sha:
        raise ValueError('Resume execution identity differs; use a new output directory')
    manifest['collection_identity_sha256'] = identity_sha
    persist()
    # Even fully completed resumes validate bytes before returning.
    pending = []
    for item in design['items']:
        if options.get('items') and item['id'] not in options['items']:
            continue
        old = manifest['items'][item['id']]
        if completed_valid(output, old, identity_sha, probe_request(design, item)):
            progress('validated_complete_on_resume', item['id'])
        else:
            if old['status'] != 'pending':
                attempt.setdefault('superseded_item_records', {})[item['id']] = old.copy()
            manifest['items'][item['id']] = dict(status='pending', design=item)
            pending.append(item)
    if not pending:
        return
    guard('weight load and native checkpoint gate', 900)
    from exp_code.baselines.model_loading import load_model
    from exp_code.common.sc_sampler import SCSampler
    from exp_code.semantic_intervention.debug_console import verify_checkpoint, prepare_canvas
    from .batch_gates import verify_mbpp, breakpoint_identity, tree_identity, compare_events
    from .resident_position import probe
    import dialog
    import jax
    progress('SINGLE_WEIGHT_LOAD_START')
    model, params, tokenizer, _ = load_model(Path(previous['checkpoint']), 1)
    memory('model_loaded')
    chat = SCSampler(model=model, params=params, tokenizer=tokenizer, cache_length=previous['cache_length'],
        max_out_length=previous['max_new_tokens'], canvas_length=design['canvas_length'],
        max_denoising_steps=design['max_denoising_steps'])
    batch, focus, seed = validate_design(design, previous)
    prompts = [dialog.Conversation(dialog.User(c['prompt'])).as_text(format=tokenizer.FORMAT) for c in batch]
    tokens = list(dict.fromkeys([i['token'] for i in design['items']] + design['neighbors']))
    inputs = chat._get_inputs(prompt=prompts, images=None, add_bos=True, has_batch_dim=True, sharding=None)
    write_json(attempt_dir/'case.json', dict(case_id=design['case_id'], batch=batch, focus=focus, seed=seed,
        rendered_prompts=prompts, tokenized_input_identity=tree_identity(inputs),
        tokens={str(t): dict(decoded=tokenizer.decode([t])) for t in tokens},
        sampler=dict(canvas_length=chat.canvas_length, max_denoising_steps=chat.max_denoising_steps,
            static_max_out_length=chat.max_out_length, cache_length=chat.cache_length,
            stopping=repr(chat.early_stop_fn), logit_shaper=repr(chat.logit_shaper), sampling=repr(chat.sampling),
            sc_mode=repr(chat.sc_mode), selection=repr(chat.sample_from_predictions))))
    del inputs
    native_dir = attempt_dir/'native_gate'
    native_dir.mkdir()
    guard('existing GSM480 checkpoint gate', 720)
    progress('native_gate_start')
    memory('native_gate_before')
    gate_started = time.monotonic()
    # Direct verifier propagates; resident_checkpoint_gate and Console are deliberately unused.
    gate_result = verify_checkpoint(chat, previous, native_dir)
    memory('native_gate_saved_before_release', gate_result[0])
    if cleanup:
        release_engine(gate_result[0])
    del gate_result
    gc.collect()
    memory('native_gate_after_release')
    attempt['native_gate'] = dict(status='passed', path=str(native_dir.relative_to(output)), seconds=time.monotonic()-gate_started)
    progress('native_gate_passed', attempt['native_gate'])
    progress('mbpp_gate_start')
    mbpp = verify_mbpp(chat, prompts, seed, focus, attempt_dir/'mbpp_gate', guard,
                       memory=memory, cleanup=cleanup)
    gc.collect()
    memory('mbpp_gate_after_release')
    attempt['mbpp_gate'] = dict(status=mbpp['status'], path=str((attempt_dir/'mbpp_gate').relative_to(output)))
    progress('mbpp_gate_passed', attempt['mbpp_gate'])
    expected_breakpoint = read_json(attempt_dir/'mbpp_gate/prefill_2_full_identity.json')
    import numpy as np
    with np.load(attempt_dir/'mbpp_gate/native_events.npz', allow_pickle=False) as archive:
        native_events = [{k: archive[k][i] for k in archive.files} for i in range(len(archive['step']))]
    # Fresh original batch/seed per item. Never load a CSV as state or reuse an edited state.
    factor = design['walltime']['timing_safety_factor']
    fresh_seconds = max(e['seconds'] for e in mbpp['execution_path'] if e['operation'].startswith('fresh_prefill'))
    step_seconds = mbpp['continuation_seconds']/max(1, mbpp['continuation_steps'])
    # Includes a fresh prefill, up to two extra breakpoint steps, 25 branch
    # steps and exports. This is measured admission guidance, not a time cap.
    estimate = max(design['walltime']['probe_floor_seconds'],
                   factor*(fresh_seconds + 27*step_seconds + 30))
    progress('measured_probe_estimate', dict(estimate_seconds=estimate,
        fresh_prefill_to_2_seconds=fresh_seconds, continuation_seconds_per_step=step_seconds))
    for item in pending:
        guard('prefill and probe '+item['id'], estimate)
        started = time.monotonic()
        item_dir = attempt_dir/'items'/item['id']
        item_dir.mkdir(parents=True)
        record = manifest['items'][item['id']]
        record.update(status='running', attempt=attempt['id'], identity_sha256=identity_sha,
                      output_path=str(item_dir.relative_to(output)))
        progress('item_start', item)
        memory('config_'+item['id']+'_start')
        try:
            engine, initial = prepare_canvas(chat, prompts, seed)
            memory('config_'+item['id']+'_prefilled', engine)
            engine.until(2)
            memory('config_'+item['id']+'_t2', engine)
            fresh_identity = breakpoint_identity(engine, initial)
            write_json(item_dir/'fresh_t2_identity.json', fresh_identity)
            if fresh_identity != expected_breakpoint:
                raise AssertionError('New item fresh MBPP breakpoint differs from passed reproducibility gate')
            engine.until(item['step'])
            differences = compare_events(native_events[:engine.state.step], engine.recorder.events)
            write_json(item_dir/'baseline_native_trace_differences.json', differences)
            if differences:
                raise AssertionError('New item unchanged breakpoint trace differs from matched native gate')
            eligible = eligibility(engine, design, item, focus)
            record['eligibility'] = eligible
            write_json(item_dir/'eligibility.json', eligible)
            write_json(item_dir/'breakpoint_identity.json', breakpoint_identity(engine, initial))
            progress('eligibility', dict(item=item['id'], **eligible))
            if not eligible['eligible']:
                record['status'] = 'skipped'
                progress('item_skipped', dict(item=item['id'], reasons=eligible['reasons']))
                continue
            guard('probe '+item['id'], max(design['walltime']['probe_floor_seconds'], estimate-(time.monotonic()-started)))
            result = probe(engine, source=design['source'], target=design['target'], token=item['token'],
                fraction=item['fraction'], horizon=design['horizon'], batch=focus, case_id=design['case_id'],
                seed=seed, tokenizer=tokenizer, output=item_dir/'probe', neighbors=design['neighbors'],
                memory=memory, host_controls=cleanup)
            validate_probe(item_dir/'probe')
            record.update(status='complete', probe_path=str((item_dir/'probe').relative_to(output)),
                report=result, branch_paths={b: str((item_dir/'probe'/b).relative_to(output)) for b in BRANCHES},
                probe_request=probe_request(design, item),
                native_gate_path=str((native_dir/'checkpoint_gate.json').relative_to(output)),
                mbpp_gate_path=str((attempt_dir/'mbpp_gate/gate.json').relative_to(output)),
                gate_artifacts={str(p.relative_to(output)): sha256(p) for p in (
                    native_dir/'checkpoint_gate.json', attempt_dir/'mbpp_gate/gate.json', attempt_dir/'identity.json',
                    attempt_dir/'case.json', item_dir/'breakpoint_identity.json', item_dir/'eligibility.json',
                    item_dir/'fresh_t2_identity.json', item_dir/'baseline_native_trace_differences.json')},
                artifacts={str(p.relative_to(item_dir/'probe')): sha256(p) for p in sorted((item_dir/'probe').rglob('*')) if p.is_file()},
                seconds=time.monotonic()-started)
            estimate = max(estimate, factor*record['seconds'])
            progress('item_complete', dict(item=item['id'], seconds=record['seconds'],
                stops={b: result['branches'][b]['completed_step'] for b in BRANCHES}))
        except BaseException as error:
            record.update(status='remaining' if isinstance(error, TimeLimit) else 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error))
            write_json(item_dir/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
            persist()
            raise
        finally:
            if 'engine' in locals():
                memory('config_'+item['id']+'_saved_before_release', engine)
                if cleanup:
                    release_engine(engine)
                del engine
            if 'initial' in locals():
                del initial
            gc.collect()
            memory('config_'+item['id']+'_after_release')
            persist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/direction2_pilot.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--items', nargs='+', help='Select existing frozen item IDs; no new conditions')
    parser.add_argument('--memory-policy', choices=('observe','cleanup'), default='cleanup')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        with process_lock(output):
            return run_locked(args, output)
    except Exception as error:
        print('DIRECTION2_FATAL', repr(error), file=sys.stderr, flush=True)
        return 1


def run_locked(args, output):
    design = read_json(args.config)
    previous = read_json(ROOT/design['run_config'])
    validate_design(design, previous)
    selected = args.items or [i['id'] for i in design['items']]
    if len(set(selected)) != len(selected) or not set(selected) <= {i['id'] for i in design['items']}:
        raise ValueError('Selected items must be unique IDs in the frozen pilot')
    execution_options = dict(items=selected, memory_policy=args.memory_policy)
    sources = source_files()
    env = environment()
    base = dict(source_files=sources, source_sha256=digest(sources), design=design, run_config=previous,
        design_file_sha256=sha256(args.config), run_config_file_sha256=sha256(ROOT/design['run_config']),
        python=env['python'], executable=env['executable'], dependencies=env['dependencies'], flags=env['flags'],
        module_origins=env['module_origins'])
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            raise ValueError('Output already has a manifest; use --resume or a new directory')
        manifest = read_json(manifest_path)
        if manifest['base_identity'] != base:
            raise ValueError('Resume source/config/environment identity mismatch; old data preserved; use a new directory')
        if manifest.get('execution_options') != execution_options:
            raise ValueError('Resume selected items or memory policy differs')
        # A killed process may have left running entries; retain them, never mark complete.
        for old in manifest['attempts']:
            if old['status'] == 'running':
                old.update(status='interrupted', error='Previous process ended without a final status')
    else:
        if args.resume:
            raise ValueError('--resume requires an existing manifest, not just a directory')
        if any(p.name != '.runner.lock' for p in output.iterdir()):
            raise ValueError('Nonempty output without a manifest; preserve it and choose a new directory')
        manifest = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(),
            base_identity=base, base_identity_sha256=digest(base), status='prepared', attempts=[],
            execution_options=execution_options,
            items={i['id']: dict(status='pending', design=i) for i in design['items']})
        # Written before model loading or any new intervention outcomes.
        write_json(output/'frozen_design.json', design)
        write_json(output/'run_config.json', previous)
    attempt_id = str(time.time_ns()) + '-' + str(os.getpid())
    attempt_dir = output/'attempts'/attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempt = dict(id=attempt_id, path=str(attempt_dir.relative_to(output)), status='running',
                   environment=env, progress=[], started_utc=datetime.now(timezone.utc).isoformat())
    manifest['attempts'].append(attempt)
    manifest['status'] = 'running'
    write_json(manifest_path, manifest)
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('SIGTERM: preserve attempt and restore active probe')
    signal.signal(signal.SIGTERM, interrupted)
    code = 0
    try:
        write_json(attempt_dir/'source_hashes.json', sources)
        with tarfile.open(attempt_dir/'source.tar.gz', 'w:gz') as archive:
            for relative in sources:
                archive.add(ROOT/relative, arcname=relative)
        write_json(attempt_dir/'git.json', dict(head=command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
            status=command(['git', '-C', str(ROOT), 'status', '--short'])))
        execute(design, previous, output, manifest, attempt)
        attempt['status'] = 'complete'
        manifest['status'] = 'complete'
    except BaseException as error:
        status, code = (('gpu_blocked', 2) if isinstance(error, GPUUnavailable) else
                       ('walltime_deferred', 3) if isinstance(error, TimeLimit) else
                       ('interrupted', 130) if isinstance(error, KeyboardInterrupt) else ('failed', 1))
        attempt.update(status=status, error=repr(error))
        manifest['status'] = status
        write_json(attempt_dir/'failure.json', dict(status=status, error=repr(error), traceback=traceback.format_exc()))
        print('DIRECTION2_STOP', status, repr(error), flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        attempt['ended_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, manifest)
        write_json(output/'resume_plan.json', dict(status=manifest['status'],
            completed=[k for k,v in manifest['items'].items() if v['status']=='complete'],
            skipped=[k for k,v in manifest['items'].items() if v['status']=='skipped'],
            remaining=[k for k,v in manifest['items'].items() if k in selected and v['status'] not in ('complete','skipped')],
            command=shlex.join([sys.executable, '-u', '-m', 'exp_code.positional_intervention.run_batch',
                '--config', str(args.config.resolve()), '--output', str(output), '--resume',
                '--memory-policy', args.memory_policy, '--items', *selected]),
            note='Enter an allocated GPU shell with the required flags. Re-run all gates and fresh prefills. Skipped items are rechecked on resume; old attempts remain.'))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
