"""Inspect checkpoint memory, then validate one unchanged native MBPP canvas.

No quantization, dtype conversion, alternate backend, or intervention edits.
An inadequate device is rejected before restoring any weight payload.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from .analyze import read_json, sha256, write_json
from .run_batch import ROOT, preflight_gpu, scheduler, validate_design, checkpoint_identity, command, environment

# Match the pinned native _remove_mm_params. _partial_restore currently restores
# these tensors and deletes them afterwards; they therefore count at load peak.
MM_EMBEDDER = {
    'mm_input_projection', 'mm_soft_embedding_norm', 'mm_pre_projection_norm',
    'mm_input_embedding_extra', 'audio_input_projection', 'audio_soft_embedding_norm',
    'audio_input_embedding', 'audio_input_embedding_extra',
}


class InsufficientMemory(RuntimeError):
    pass


def tensor_record(parts, metadata):
    dtype = metadata['dtype']
    itemsize = 2 if dtype == 'bfloat16' else np.dtype(dtype).itemsize
    shape = metadata['shape']
    if not all(isinstance(n, int) and n > 0 for n in shape):
        raise ValueError('Invalid checkpoint shape')
    elements = math.prod(shape)
    multimodal = parts[0] in ('vision_encoder', 'audio_encoder') or (
        parts[0] == 'embedder' and len(parts) > 1 and parts[1] in MM_EMBEDDER)
    return dict(name='.'.join(parts), shape=shape, dtype=dtype, itemsize=itemsize,
                elements=elements, bytes=elements*itemsize, retained_text_only=not multimodal)


def summarize_tensors(rows):
    return dict(tensor_count=len(rows), total_elements=sum(r['elements'] for r in rows),
        total_restore_bytes=sum(r['bytes'] for r in rows),
        retained_text_bytes=sum(r['bytes'] for r in rows if r['retained_text_only']),
        temporary_multimodal_bytes=sum(r['bytes'] for r in rows if not r['retained_text_only']),
        dtype_counts={d: sum(r['dtype'] == d for r in rows) for d in sorted({r['dtype'] for r in rows})},
        assumption='Pinned native loader restores all leaves before _release_skip deletes multimodal arrays; no tensor payload read by inspection.')


def inspect_checkpoint(checkpoint, output):
    import tensorstore as ts
    checkpoint = checkpoint.resolve()
    tree = read_json(checkpoint/'_METADATA')['tree_metadata']
    store = ts.KvStore.open({'driver': 'ocdbt', 'base': {'driver': 'file', 'path': str(checkpoint)+'/'}}).result()
    rows, raw = [], {}
    for node in tree.values():
        parts = [part['key'] for part in node['key_metadata']]
        name = '.'.join(parts)
        metadata = json.loads(bytes(store.read(name+'/.zarray').result().value))
        rows.append(tensor_record(parts, metadata))
        raw[name] = metadata
    summary = summarize_tensors(rows)
    summary.update(checkpoint=str(checkpoint), metadata_sha256=sha256(checkpoint/'_METADATA'),
        native_loader_sha256=sha256(ROOT/'dg/gemma/gm/ckpts/_checkpoint.py'))
    write_json(output/'checkpoint_tensor_metadata.json', raw)
    write_json(output/'weight_memory.json', summary)
    with (output/'checkpoint_tensors.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def require_capacity(summary, physical_bytes, allocator_limit):
    required = summary['total_restore_bytes']
    if allocator_limit is None:
        raise InsufficientMemory('JAX did not expose its allocator byte limit; loading was not attempted')
    if required >= physical_bytes or required >= allocator_limit:
        raise InsufficientMemory(
            f'Unchanged restore requires at least {required} bytes ({required/2**30:.3f} GiB); '
            f'physical GPU {physical_bytes/2**30:.3f} GiB, allocator limit {allocator_limit/2**30:.3f} GiB. '
            'This lower bound excludes loading workspace, KV cache and generation activations.')


def validate(args, design, previous, output, report):
    # This preflight rejects competing GPU processes and verifies CUDA default
    # plus CPU callbacks, without touching any unrelated process.
    hardware = preflight_gpu()
    report['hardware'] = hardware
    report['environment'] = environment()
    selector = ['--id=' + os.environ['CUDA_VISIBLE_DEVICES']] if os.environ.get('CUDA_VISIBLE_DEVICES') else []
    report['assigned_gpu_memory'] = command(['nvidia-smi', *selector,
        '--query-gpu=name,uuid,driver_version,memory.total,memory.free,memory.used,pci.bus_id,compute_cap',
        '--format=csv'])
    import jax
    from jaxlib import xla_client
    device = jax.devices()[0]
    report['jax_allocated_device'] = dict(device=str(device), kind=device.device_kind,
        local_hardware_id=getattr(device, 'local_hardware_id', None),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    assigned_name = hardware['nvidia_smi'].splitlines()[0].split(',')[0].strip()
    if device.device_kind != assigned_name:
        raise RuntimeError('JAX GPU model differs from the allocated device reported by nvidia-smi')
    report['allocator_options'] = xla_client.generate_pjrt_gpu_plugin_options()
    report['memory_before_restore'] = jax.devices()[0].memory_stats()
    physical_mib = float(hardware['nvidia_smi'].splitlines()[0].split(',')[3].strip().split()[0])
    report['physical_memory_bytes'] = int(physical_mib*2**20)
    write_json(output/'loading_validation.json', report)
    require_capacity(report['weight_memory'], report['physical_memory_bytes'],
                     report['memory_before_restore'].get('bytes_limit'))
    if physical_mib < args.minimum_gpu_mib:
        raise InsufficientMemory(f'GPU has {physical_mib} MiB; launch requires at least {args.minimum_gpu_mib} MiB')
    if scheduler()['remaining_seconds'] < 1020:
        raise RuntimeError('Less than 900 seconds estimated validation plus 120 seconds to save; no restore started')
    report['checkpoint_identity'] = checkpoint_identity(previous['checkpoint'],
        lambda operation, detail: print(operation, json.dumps(detail), flush=True))
    if scheduler()['remaining_seconds'] < 1020:
        raise RuntimeError('Insufficient scheduler time after checkpoint verification; no restore started')
    from exp_code.baselines.model_loading import load_model
    from exp_code.common.sc_sampler import SCSampler
    import dialog
    print('VALIDATION_SINGLE_WEIGHT_LOAD_START', flush=True)
    report['weight_load_attempts'] = 1
    write_json(output/'loading_validation.json', report)
    started = time.monotonic()
    model, params, tokenizer, _ = load_model(Path(previous['checkpoint']), 1)
    jax.block_until_ready(params)
    leaves = jax.tree.leaves(params)
    logical_bytes = sum(a.size*a.dtype.itemsize for a in leaves)
    report.update(load_seconds=time.monotonic()-started, actual_parameter_bytes=logical_bytes,
        parameter_arrays=len(leaves), unique_parameter_array_objects=len({id(a) for a in leaves}),
        actual_parameter_dtypes=sorted({str(a.dtype) for a in leaves}),
        memory_after_restore=jax.devices()[0].memory_stats())
    if logical_bytes != report['weight_memory']['retained_text_bytes']:
        raise AssertionError('Restored parameter byte count does not match checkpoint metadata')
    report['load_passed'] = True
    write_json(output/'loading_validation.json', report)
    if scheduler()['remaining_seconds'] < 420:
        raise RuntimeError('Not enough remaining scheduler time for minimal generation and saving')
    batch, focus, seed = validate_design(design, previous)
    chat = SCSampler(model=model, params=params, tokenizer=tokenizer,
        cache_length=previous['cache_length'], max_out_length=previous['max_new_tokens'],
        canvas_length=design['canvas_length'], max_denoising_steps=design['max_denoising_steps'])
    prompts = [dialog.Conversation(dialog.User(c['prompt'])).as_text(format=tokenizer.FORMAT) for c in batch]
    report['generation'] = dict(case_id=design['case_id'], focus=focus, seed=seed,
        batch=batch, rendered_prompts=prompts, max_new_tokens=256, static_max_out_length=chat.max_out_length,
        canvas_length=chat.canvas_length, max_denoising_steps=chat.max_denoising_steps,
        stopping=repr(chat.early_stop_fn), logit_shaper=repr(chat.logit_shaper), sc_mode=repr(chat.sc_mode),
        scope='One unchanged native canvas for the original four-member batch; operational smoke check, not parity or accuracy.')
    write_json(output/'loading_validation.json', report)
    print('VALIDATION_NATIVE_MBPP_CANVAS_START', flush=True)
    started = time.monotonic()
    result = chat.sample(prompts, rng=seed, max_new_tokens=256, return_state=True)
    jax.block_until_ready(result.tokens)
    jax.effects_barrier()
    tokens = np.asarray(result.tokens)
    np.save(output/'native_tokens.npy', tokens)
    write_json(output/'decoded_canvas.json', {c['id']: tokenizer.decode(tokens[i, :256]) for i,c in enumerate(batch)})
    report.update(generation_seconds=time.monotonic()-started, generation_passed=True,
        memory_after_generation=jax.devices()[0].memory_stats(), status='passed',
        suite_requires_original_native_and_mbpp_gates=True)
    print('LOADING_AND_MINIMAL_GENERATION_PASSED', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/direction2_pilot.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true', help='Read shape/dtype metadata only; do not initialize JAX or load tensors')
    parser.add_argument('--minimum-gpu-mib', type=int, default=0)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', created_utc=datetime.now(timezone.utc).isoformat(),
        weight_load_attempts=0, load_passed=False, generation_passed=False,
        versions={n: importlib.metadata.version(n) for n in ('jax','jaxlib','orbax-checkpoint','gemma')})
    try:
        design = read_json(args.config)
        previous = read_json(ROOT/design['run_config'])
        validate_design(design, previous)
        report['config_sha256'] = sha256(args.config)
        report['weight_memory'] = inspect_checkpoint(Path(previous['checkpoint']), args.output)
        if args.inspect_only:
            report['status'] = 'inspected_only'
        else:
            validate(args, design, previous, args.output, report)
        return 0
    except BaseException as error:
        report.update(status='insufficient_gpu_memory' if isinstance(error, InsufficientMemory) else 'failed',
                      error=repr(error), traceback=traceback.format_exc())
        print('LOADING_VALIDATION_STOP', report['status'], repr(error), flush=True)
        return 2 if isinstance(error, InsufficientMemory) else 1
    finally:
        write_json(args.output/'loading_validation.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
