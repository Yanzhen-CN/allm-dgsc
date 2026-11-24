"""Synchronized memory evidence and disposal of finished debugger resources.

No array donation, allocator changes, native-loop changes, or parameter deletion.
"""
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np


def tree_bytes(tree):
    arrays = {id(a): a for a in jax.tree.leaves(tree) if hasattr(a, 'nbytes')}
    return sum(a.nbytes for a in arrays.values())


def state_signature(state, chunk_bytes=8*1024*1024):
    """Full bitwise state identity with bounded host transfers; retain no arrays."""
    result = {'step': state.step, 'fields': {}}
    for field in ('canvas', 'sc_embeddings', 'sc_logits', 'done', 'rng'):
        value = getattr(state, field)
        if field == 'rng':
            value = jax.random.key_data(value)
        if value is None:
            result['fields'][field] = None
            continue
        flat = value.reshape(-1)
        count = max(1, chunk_bytes//value.dtype.itemsize)
        digest = hashlib.sha256()
        for start in range(0, value.size, count):
            digest.update(np.asarray(flat[start:start+count]).tobytes())
        result['fields'][field] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest.hexdigest())
    return result


def release_engine(engine):
    """Only after required artifacts are saved; shared model arrays stay alive."""
    if engine is None:
        return
    jax.effects_barrier()
    # Compiled host callbacks close over this debugger. Deleting the Python
    # variable + gc.collect alone does not break that executable/callback cycle.
    for name in ('_advance_native', '_sample_step', '_swap_and_encode',
                 '_mix_and_encode', '_position_encode'):
        fn = getattr(engine, name, None)
        if fn is not None and hasattr(fn, 'clear_cache'):
            fn.clear_cache()
        if hasattr(engine, name):
            setattr(engine, name, None)
    engine.snapshots.clear()
    engine.recorder.events.clear()
    engine.recorder.final_canvas.clear()
    engine._captured_logits = None
    for name in ('state', 'cache', 'params', 'sampler', 'full_attention_mask',
                 'positions', 'attention', 'sliding', 'noises'):
        setattr(engine, name, None)


class MemoryAudit:
    def __init__(self, path, enabled=True):
        self.path = Path(path)
        self.enabled = enabled

    def __call__(self, label, engine=None, retained=None):
        if not self.enabled:
            return
        event = dict(utc=datetime.now(timezone.utc).isoformat(), label=label)
        # A failed asynchronous dispatch can poison subsequent barriers. Retain
        # that failure explicitly; never report such a reading as synchronized.
        try:
            if engine is not None and engine.state is not None:
                jax.block_until_ready((engine.state.canvas, engine.state.sc_logits, engine.cache))
            jax.effects_barrier()
            event['synchronized'] = True
        except Exception as error:
            event.update(synchronized=False, synchronization_error=repr(error))
        try:
            device = jax.devices()[0]
            event['device_memory'] = device.memory_stats()
            live = jax.live_arrays()
            event['live_arrays'] = dict(count=len(live), logical_bytes=sum(a.nbytes for a in live))
            del live
            from exp_code.common.canvas_debugger import CanvasDebugger
            engines = [o for o in gc.get_objects() if isinstance(o, CanvasDebugger)]
            event['debuggers'] = [dict(id=id(e), step=getattr(e.state, 'step', None),
                cache_bytes=tree_bytes(e.cache), state_bytes=tree_bytes(vars(e.state)) if e.state is not None else 0,
                snapshots=len(e.snapshots), captured_host_logits_bytes=tree_bytes(e._captured_logits),
                recorder_host_bytes=tree_bytes(e.recorder.events),
                jit_cache_sizes={n:getattr(e,n)._cache_size() for n in ('_advance_native','_position_encode')
                    if getattr(e,n,None) is not None and hasattr(getattr(e,n),'_cache_size')}) for e in engines]
            del engines
            event['retained_control_bytes'] = {k:tree_bytes(vars(v)) for k,v in (retained or {}).items() if v is not None}
            client = device.client
            if hasattr(client, 'live_executables'):
                executables = client.live_executables()
                event['executables'] = []
                for exe in executables:
                    data = {}
                    if hasattr(exe, 'size_of_generated_code_in_bytes'):
                        data['generated_code_bytes'] = exe.size_of_generated_code_in_bytes()
                    if hasattr(exe, 'get_compiled_memory_stats'):
                        stats = exe.get_compiled_memory_stats()
                        data.update({key:getattr(stats,key) for key in ('argument_size_in_bytes',
                            'output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes') if hasattr(stats,key)})
                    event['executables'].append(data)
                event['executable_count'] = len(executables)
                del executables
            if device.platform == 'gpu':
                import os
                gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
                result = subprocess.run(['nvidia-smi', *(['--id='+gpu] if gpu else []),
                    '--query-gpu=uuid,memory.total,memory.free,memory.used,utilization.gpu', '--format=csv'],
                    text=True,capture_output=True,timeout=15)
                event['nvidia_smi'] = result.stdout.strip()
        except Exception as error:
            event['measurement_error'] = repr(error)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        stats = event.get('device_memory') or {}
        print('DIRECTION2_MEMORY', label, json.dumps(dict(bytes_in_use=stats.get('bytes_in_use'),
            peak_bytes_in_use=stats.get('peak_bytes_in_use'), pool_bytes=stats.get('pool_bytes'),
            debugger_count=len(event.get('debuggers',[])), synchronized=event.get('synchronized'))), flush=True)
