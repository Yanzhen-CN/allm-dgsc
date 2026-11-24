"""Position edits and bounded paired probes using the validated native engine."""
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .position_math import entropy, probabilities, relocate_rows
from .analyze import write_json


def move(engine, source, target, token, fraction, batch):
    if source == target:
        raise ValueError('Source and destination must differ')
    engine._editable(batch, source)
    engine._editable(batch, target)
    state = engine.state
    for edit in engine.edits:
        if (edit.get('after_step_number') == state.step
                and edit.get('kind') == 'sc_embedding_zero'
                and edit.get('batch') == batch
                and edit.get('position') in (source, target)):
            raise ValueError('Restore before posmove: a touched row has a zero-SC override')
    indices = jnp.array([source, target])
    old_rows = np.asarray(state.sc_logits[batch, indices], dtype=np.float32)
    requested_rows, details = relocate_rows(old_rows, token, fraction)
    before = probabilities(old_rows)
    if not details['exact_noop']:
        if not hasattr(engine, '_position_encode'):
            model, embedding_dtype = engine.sampler.model, state.sc_embeddings.dtype
            def encode(params, logits):
                result = model.apply(
                    {'params': params}, logits,
                    method=lambda model, x: model.embedder.encode_logits(x))
                return result.astype(embedding_dtype)
            engine._position_encode = jax.jit(encode)
        logits = state.sc_logits.at[batch, indices].set(
            jnp.asarray(requested_rows, dtype=state.sc_logits.dtype))
        encoded = engine._position_encode(engine.params, logits)
        embeddings = state.sc_embeddings.at[batch, indices].set(encoded[batch, indices])
        jax.block_until_ready(embeddings)
        engine.state = dataclasses.replace(state, sc_logits=logits, sc_embeddings=embeddings)
    after = probabilities(np.asarray(engine.state.sc_logits[batch, indices], dtype=np.float32))
    record = dict(kind='sc_posmove', after_step_number=state.step,
        first_affected_step_number=state.step+1, batch=batch,
        source_position=source, target_position=target, token_id=token,
        **details, actual_anchor_after=after[:, token].tolist(),
        actual_source_loss=float(before[0, token]-after[0, token]),
        actual_destination_gain=float(after[1, token]-before[1, token]),
        actual_anchor_mass_error=float(after[:, token].sum()-before[:, token].sum()),
        requested_vs_stored_probability_l1=np.abs(after-probabilities(requested_rows)).sum(axis=1).tolist(),
        entropy_before=[entropy(row) for row in before],
        entropy_after=[entropy(row) for row in after],
        embedding_effects=[engine._embedding_effect(state.sc_embeddings[batch, pos],
            engine.state.sc_embeddings[batch, pos]) for pos in (source, target)],
        canvas_rng_done_step_cache_unchanged=True,
        probability_note='Full-vocabulary softmax of stored logits, evaluated in float64 on host; BF16 rounding reported')
    engine.edits.append(record)
    return record


def observe(engine, batch, start, end, tokens, phase):
    """Inclusive window. Probability, sample and accepted-canvas status differ."""
    state = engine.state
    if state.sc_logits is None:
        raise ValueError('Run at least one step first')
    if not 0 <= start <= end < engine.sampler.canvas_length:
        raise ValueError('Invalid inclusive position window')
    if not tokens or len(set(tokens)) != len(tokens) or any(
            not 0 <= token < engine.sampler.text_vocab_size for token in tokens):
        raise ValueError('Use unique valid token IDs')
    probs = probabilities(np.asarray(state.sc_logits[batch, start:end+1], dtype=np.float32))
    event = engine.recorder.events[-1] if engine.recorder.events else None
    active = bool(event['active'][batch]) if event is not None else False
    records = []
    summary = []
    canvas = np.asarray(state.canvas[batch])
    for token in tokens:
        values = probs[:, token]
        mass = float(values.sum())
        summary.append(dict(token_id=token, local_mass=mass,
            local_centroid=float(np.dot(np.arange(start, end+1), values)/mass)
                if mass > 1e-12 else None))
        for offset, pos in enumerate(range(start, end+1)):
            records.append(dict(completed_step=state.step, phase=phase,
                position=pos, token_id=token, probability=float(values[offset]),
                sc_argmax_token=int(probs[offset].argmax()),
                canvas_token_id=int(canvas[pos]), active_at_last_step=active,
                accepted_at_last_step=bool(event['accepted'][batch, pos]) if active else None,
                sampled_token_id=int(event['sampled'][batch, pos]) if active else None,
                focus_stopped=bool(state.done[batch])))
    return dict(completed_step=state.step, phase=phase, window_inclusive=[start, end],
                focus_stopped=bool(state.done[batch]), records=records, summary=summary)


def _equal_states(first, second):
    if first.step != second.step:
        return False
    for field in ('canvas', 'sc_embeddings', 'sc_logits', 'done'):
        if not bool(jnp.array_equal(getattr(first, field), getattr(second, field))):
            return False
    return bool(jnp.array_equal(jax.random.key_data(first.rng), jax.random.key_data(second.rng)))


def _isolation(before, after, batch, source, target, cache, engine):
    if (before.step != after.step or before.canvas is not after.canvas
            or before.rng is not after.rng or before.done is not after.done
            or cache is not engine.cache):
        raise AssertionError('Position edit altered canvas/RNG/done/step/cache')
    # Mask only the two edited rows; compare all other rows, not just the focus.
    mask = jnp.ones(before.sc_logits.shape[:2], dtype=bool).at[batch, source].set(False).at[batch, target].set(False)
    for field in ('sc_logits', 'sc_embeddings'):
        a, b = getattr(before, field), getattr(after, field)
        if not bool(jnp.all(jnp.where(mask[..., None], a == b, True))):
            raise AssertionError('Position edit changed an unedited '+field+' row')


def _event_digest(events):
    digest = hashlib.sha256()
    for event in events:
        for key in sorted(event):
            value = np.asarray(event[key])
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def probe(engine, *, source, target, token, fraction, horizon, batch,
          case_id, seed, tokenizer, output, neighbors=(), memory=None, host_controls=False):
    """Five paired branches; validation failures mark all data invalid.

    Baseline, exact no-op, forward shift, reverse shift, repeated forward.
    Every branch restores the same state and native random stream. Advance
    at most horizon steps; never clear done or extend the original cap.
    """
    if not 1 <= horizon <= 8:
        raise ValueError('Use 1..8 continuation steps for this bounded pilot')
    if not 0 < fraction <= 1 or not np.isfinite(fraction):
        raise ValueError('Probe fraction must be in (0,1]')
    engine._editable(batch, source)
    engine._editable(batch, target)
    if source == target:
        raise ValueError('Source and destination must differ')
    start, end = max(0, min(source, target)-6), min(engine.sampler.canvas_length-1, max(source, target)+6)
    tokens = list(dict.fromkeys([token, *neighbors]))
    observe(engine, batch, start, end, tokens, 'before_edit')  # Validate before creating files.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = engine.state
    cache = engine.cache
    base_trace_digest = _event_digest(engine.recorder.events)
    old_limit = engine.max_snapshots
    name = '__position_probe_'+str(time.time_ns())
    engine.max_snapshots = max(old_limit, len(engine.snapshots)+1)
    engine.save(name)
    engine.max_snapshots = old_limit
    report = dict(status='running', case_id=case_id, seed=seed, focus=batch,
        base_completed_step=base.step, source_position=source, target_position=target,
        anchor_token_id=token, fraction=fraction, horizon=horizon,
        max_denoising_steps=engine.sampler.max_denoising_steps,
        window_inclusive=[start, end], watched_tokens=tokens,
        stopping_policy='native adaptive stopping; canvas0 only',
        interpretation='Exploratory paired case, not independent samples or final MBPP correctness',
        control_state_storage='full bitwise SHA256 identities, 8 MiB host chunks' if host_controls else 'GPU states',
        checks={}, branches={})
    write_json(output/'position_report.json', report)
    measure = memory or (lambda *args, **kwargs: None)
    from .memory_audit import state_signature
    try:
        baseline = forward = None
        baseline_digest = forward_digest = None
        for condition, src, dst, amount in (
            ('baseline', source, target, None), ('noop', source, target, 0.),
            ('forward', source, target, fraction), ('reverse', target, source, fraction),
            ('repeat_forward', source, target, fraction)):
            engine.restore(name)
            measure('branch_'+condition+'_restored', engine, {'baseline':baseline,'forward':forward} if not host_controls else {})
            observations = [observe(engine, batch, start, end, tokens, 'before_edit')]
            edit = None
            if amount is not None:
                edit = move(engine, src, dst, token, amount, batch)
                _isolation(base, engine.state, batch, src, dst, cache, engine)
                if amount == 0 and engine.state is not base:
                    raise AssertionError('No-op did not retain the exact state object')
                observations.append(observe(engine, batch, start, end, tokens, 'after_edit'))
            measure('branch_'+condition+'_before_advance', engine)
            for _ in range(horizon):
                if engine.finished or bool(engine.state.done[batch]):
                    break
                engine.step()
                observations.append(observe(engine, batch, start, end, tokens, 'next_model_prediction'))
            folder = output/condition
            folder.mkdir()
            branch = engine.recorder.export(folder, batch, case_id, seed, tokenizer)
            focus_done = bool(engine.state.done[batch])
            reason = ('step_cap' if engine.state.step >= engine.sampler.max_denoising_steps
                      else 'adaptive' if focus_done else 'probe_horizon')
            branch.update(edits=engine.edits, scope='canvas0 only',
                partial=not (focus_done or engine.finished),
                stop_reasons=json.dumps([reason]),
                completed_step=engine.state.step, focus_stopped=bool(engine.state.done[batch]),
                decoded_canvas=tokenizer.decode(engine.state.canvas[batch]))
            (folder/'branch.json').write_text(json.dumps(branch, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            (folder/'position_trace.json').write_text(json.dumps(observations, indent=2)+'\n', encoding='utf-8')
            import csv
            records = [row for observation in observations for row in observation['records']]
            with (folder/'position_probabilities.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            report['branches'][condition] = dict(completed_step=engine.state.step,
                focus_stopped=bool(engine.state.done[batch]), edit=edit,
                observed_followups=len([o for o in observations if o['phase']=='next_model_prediction']),
                stop_reason=reason, output_path=str(folder))
            digest = _event_digest(engine.recorder.events)
            if condition == 'baseline':
                baseline, baseline_digest = state_signature(engine.state) if host_controls else engine.state, digest
                if host_controls: write_json(folder/'full_state_identity.json', baseline)
            elif condition == 'noop':
                equal = baseline == state_signature(engine.state) if host_controls else _equal_states(baseline, engine.state)
                if not equal or baseline_digest != digest:
                    raise AssertionError('No-op continuation differs from baseline')
                report['checks']['noop_full_state_and_trace_equal'] = True
                baseline = None
            elif condition == 'forward':
                forward, forward_digest = state_signature(engine.state) if host_controls else engine.state, digest
                if host_controls: write_json(folder/'full_state_identity.json', forward)
            elif condition == 'repeat_forward':
                equal = forward == state_signature(engine.state) if host_controls else _equal_states(forward, engine.state)
                if not equal or forward_digest != digest:
                    raise AssertionError('Repeated forward branch differs')
                report['checks']['repeated_shift_full_state_and_trace_equal'] = True
            print('POSITION_BRANCH', condition, 'completed', engine.state.step, flush=True)
            write_json(output/'position_report.json', report)
            measure('branch_'+condition+'_saved', engine, {'baseline':baseline,'forward':forward} if not host_controls else {})
        report['checks']['only_two_sc_rows_edited'] = True
        report['checks']['canvas_rng_done_step_cache_unchanged_by_edits'] = True
        report['status'] = 'passed'
    except BaseException as error:
        report['status'] = 'failed_do_not_interpret'
        report['error'] = repr(error)
        raise
    finally:
        try:
            # Drain native callbacks before replacing the recorder after an
            # interrupted dispatch; a hard scheduler kill remains resumable
            # only by a fresh prefill in a new attempt.
            jax.effects_barrier()
            engine.restore(name)
            report['restored_base_state'] = engine.state is base
            report['restored_base_trace'] = _event_digest(engine.recorder.events) == base_trace_digest
            report['restored_cache_identity'] = engine.cache is cache
            if not all(report[k] for k in ('restored_base_state', 'restored_base_trace', 'restored_cache_identity')):
                raise AssertionError('Probe did not restore its original breakpoint')
            del engine.snapshots[name]
        except BaseException as error:
            report.update(status='failed_do_not_interpret', restoration_error=repr(error))
            raise
        finally:
            write_json(output/'position_report.json', report)
    return report
