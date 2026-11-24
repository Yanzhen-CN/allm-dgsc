"""Unattended MBPP reproducibility gates using the existing native while loop."""
from __future__ import annotations

import dataclasses
import gc
import hashlib
import time

import jax
import numpy as np

from exp_code.semantic_intervention.debug_console import prepare_canvas
from .analyze import write_json


def array_identity(value):
    value = np.asarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def tree_identity(tree):
    leaves, structure = jax.tree.flatten_with_path(tree)
    return dict(structure=str(structure), leaves={jax.tree_util.keystr(path): array_identity(value)
        for path, value in leaves})


def breakpoint_identity(engine, initial):
    """Hash every state/cache element, not only the watched window or top-k."""
    state = engine.state
    return dict(step=state.step, state=tree_identity(dict(canvas=state.canvas,
        sc_embeddings=state.sc_embeddings, sc_logits=state.sc_logits,
        done=state.done, rng=jax.random.key_data(state.rng))),
        cache=tree_identity(engine.cache), full_attention_mask=tree_identity(engine.full_attention_mask),
        positions=tree_identity(engine.positions), attention=tree_identity(engine.attention),
        sliding=tree_identity(engine.sliding), initial_rng=array_identity(jax.random.key_data(initial.rng)),
        initial_done=array_identity(initial.done),
        trace=tree_identity(engine.recorder.events), recorder=tree_identity(dict(
            finished=engine.recorder.finished, done=engine.recorder.done,
            final_canvas=engine.recorder.final_canvas, canvas_index=engine.recorder.canvas_index,
            previous_step=engine.recorder.previous_step)), edits=engine.edits)


def save_events(path, events):
    if not events:
        return
    np.savez_compressed(path, **{key: np.stack([event[key] for event in events])
                               for key in events[0]})


def compare_events(a, b):
    """All four members, including inactive members; ignore added context only."""
    differences = []
    if len(a) != len(b):
        differences.append(dict(field='event_count', first=len(a), second=len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        keys = (set(x) | set(y)) - {'position_offsets'}
        for key in sorted(keys):
            if key not in x or key not in y:
                differences.append(dict(completed_step=i+1, field=key, missing=True))
            elif not np.array_equal(x[key], y[key]):
                u, v = np.asarray(x[key]), np.asarray(y[key])
                differences.append(dict(completed_step=i+1, field=key,
                    unequal=int(np.count_nonzero(u != v)) if u.shape == v.shape else None,
                    first=array_identity(u), second=array_identity(v)))
    return differences


def verify_mbpp(chat, prompts, seed, focus, output, guard=lambda *args: None, *, memory=None, cleanup=False):
    """Two fresh prefills, then matched native 256-token canvas0 continuation.

    The chat's 1024-token static output buffer is identical in every prefill;
    BOTH native sample calls explicitly request only 256 generated tokens.
    No independent compiled denoising step is introduced.
    """
    from exp_code.common.acceptance_trace import AcceptanceRecorder, TracedPredictions
    from gemma.diffusion import _sampler
    from .memory_audit import release_engine
    measure = memory or (lambda *args, **kwargs: None)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', seed=seed, focus=focus, batch_size=len(prompts),
        canvas_length=chat.canvas_length, max_new_tokens=chat.canvas_length,
        static_max_out_length=chat.max_out_length, max_denoising_steps=chat.max_denoising_steps,
        stopping='unchanged native', checks={}, execution_path=[])
    write_json(output/'gate.json', report)
    try:
        first_identity = None
        for index in range(2):
            guard('MBPP fresh prefill to step 2', 300)
            started = time.monotonic()
            engine, initial = prepare_canvas(chat, prompts, seed)
            prefill_seconds = time.monotonic()-started
            engine.until(2)
            identity = breakpoint_identity(engine, initial)
            write_json(output/f'prefill_{index+1}_full_identity.json', identity)
            save_events(output/f'prefill_{index+1}_events.npz', engine.recorder.events)
            np.savez_compressed(output/f'prefill_{index+1}_watched_rows.npz',
                sc_logits=np.asarray(engine.state.sc_logits[focus, 36:50], dtype=np.float32),
                canvas=np.asarray(engine.state.canvas), done=np.asarray(engine.state.done))
            measure(f'mbpp_prefill_{index+1}_saved', engine)
            report['execution_path'].append(dict(operation=f'fresh_prefill_{index+1}',
                prefill_seconds=prefill_seconds, seconds=time.monotonic()-started))
            if index == 0:
                first_identity = identity
                if cleanup:
                    release_engine(engine)
                del engine, initial
                gc.collect()
            else:
                report['checks']['fresh_prefill_full_state_cache_trace_equal'] = identity == first_identity
                if identity != first_identity:
                    raise AssertionError('MBPP fresh prefills differ; see per-leaf state/cache and trace identities')
        # Keep the second breakpoint and resume it unchanged after native calls.
        guard('MBPP native 256-token generation', 300)
        measure('mbpp_native_before', engine)
        report['execution_path'].append(dict(operation='untraced_native_sample', max_new_tokens=chat.canvas_length))
        result = chat.sample(prompts, rng=seed, max_new_tokens=chat.canvas_length, return_state=True)
        jax.block_until_ready(result.tokens)
        native = np.array(result.tokens, copy=True)
        np.save(output/'native_tokens.npy', native)
        del result
        gc.collect()
        measure('mbpp_native_saved', engine)
        loop = chat._initialize_sampler_loop(chat.sampling)
        recorder = AcceptanceRecorder(len(prompts), loop.end_tokens, loop.max_denoising_steps)
        traced_chat = dataclasses.replace(chat, sample_from_predictions=TracedPredictions(
            entropy_bound=loop.sample_from_predictions.entropy_bound,
            text_vocab_size=loop.text_vocab_size, max_steps=loop.max_denoising_steps,
            top_k=min(16, loop.text_vocab_size), callback=recorder.record, early_stop_fn=loop.early_stop_fn))
        guard('MBPP traced native 256-token generation', 300)
        measure('mbpp_traced_before', engine)
        report['execution_path'].append(dict(operation='traced_native_sample', max_new_tokens=chat.canvas_length))
        result = traced_chat.sample(prompts, rng=seed, max_new_tokens=chat.canvas_length, return_state=True)
        jax.block_until_ready(result.tokens)
        jax.effects_barrier()
        traced = np.array(result.tokens, copy=True)
        np.save(output/'traced_native_tokens.npy', traced)
        save_events(output/'native_events.npz', recorder.events)
        del result, traced_chat
        gc.collect()
        measure('mbpp_traced_saved', engine)
        report['checks']['instrumentation_tokens_equal'] = np.array_equal(native, traced)
        if not report['checks']['instrumentation_tokens_equal']:
            raise AssertionError('MBPP native trace instrumentation changes output')
        guard('MBPP unchanged breakpoint continuation', 300)
        measure('mbpp_continuation_before', engine)
        continuation_started = time.monotonic()
        engine.run()  # Entire original batch, same schedule and native stops.
        report['continuation_seconds'] = time.monotonic()-continuation_started
        report['continuation_steps'] = engine.state.step-2
        save_events(output/'debugger_events.npz', engine.recorder.events)
        truncated, _ = _sampler._truncate_canvas_at_stop_tokens(engine.state.canvas,
            end_tokens=loop.end_tokens, canvas_length=chat.canvas_length, done=initial.done)
        np.save(output/'debugger_tokens.npy', np.asarray(truncated))
        differences = compare_events(recorder.events, engine.recorder.events)
        write_json(output/'trace_differences.json', differences)
        report['checks']['native_vs_breakpoint_full_batch_tokens_equal'] = np.array_equal(native[:, :chat.canvas_length], truncated)
        report['checks']['native_vs_breakpoint_all_events_equal'] = not differences
        report['native_steps'] = len(recorder.events)
        report['debugger_steps'] = engine.state.step
        report['focus_stop_step'] = next((e['step']+1 for e in recorder.events if e['active'][focus] and e['stop'][focus]), None)
        report['focus_stopped'] = bool(engine.state.done[focus])
        if not all(report['checks'].values()):
            raise AssertionError('MBPP matched native/breakpoint parity failed')
        report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output/'gate.json', report)
        if 'engine' in locals():
            measure('mbpp_gate_saved_before_release', engine)
            if cleanup:
                release_engine(engine)
    return report
