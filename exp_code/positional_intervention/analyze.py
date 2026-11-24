"""Validate saved positional probes and export paired, censored CPU analyses.

No sampling, imputation, top-k renormalization, or generated-code execution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

BRANCHES = ('baseline', 'noop', 'forward', 'reverse', 'repeat_forward')
CHECKS = ('noop_full_state_and_trace_equal', 'repeated_shift_full_state_and_trace_equal',
          'only_two_sc_rows_edited', 'canvas_rng_done_step_cache_unchanged_by_edits')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _without_after(trace):
    return [o for o in trace if o['phase'] != 'after_edit']


def validate_probe(folder):
    """Validate complete artifacts; this cannot reconstruct full GPU state."""
    folder = Path(folder)
    report = read_json(folder / 'position_report.json')
    require(report['status'] == 'passed', 'Probe did not pass')
    require(report.get('restored_base_state') is True, 'Restoration was not confirmed')
    require(all(report['checks'].get(c) is True for c in CHECKS), 'Missing branch gate')
    require(set(report['branches']) == set(BRANCHES), 'Incomplete branch set')
    traces, branches, arrays = {}, {}, {}
    base, horizon = report['base_completed_step'], report['horizon']
    expected_rows = {(p, t) for p in range(report['window_inclusive'][0],
                       report['window_inclusive'][1] + 1) for t in report['watched_tokens']}
    for name in BRANCHES:
        branch_dir = folder / name
        branch = branches[name] = read_json(branch_dir / 'branch.json')
        trace = traces[name] = read_json(branch_dir / 'position_trace.json')
        require(branch['case_id'] == report['case_id'] and branch['seed'] == report['seed'],
                'Foreign branch identity')
        require(branch['completed_step'] == report['branches'][name]['completed_step'] and
                branch['focus_stopped'] == report['branches'][name]['focus_stopped'], 'Stop metadata mismatch')
        follow = [o for o in trace if o['phase'] == 'next_model_prediction']
        steps = [o['completed_step'] for o in follow]
        require(steps == list(range(base + 1, branch['completed_step'] + 1)), 'Missing continuation step')
        require(0 < len(steps) <= horizon, 'Invalid observed horizon')
        require(trace[0]['completed_step'] == base and trace[0]['phase'] == 'before_edit', 'Missing breakpoint')
        require(not trace[0]['focus_stopped'], 'Stopped focus at breakpoint')
        require([o['phase'] for o in trace] == ['before_edit'] +
                ([] if name == 'baseline' else ['after_edit']) + ['next_model_prediction'] * len(steps),
                'Invalid observation phases')
        require(all(not o['focus_stopped'] for o in follow[:-1]), 'Data after focus stop')
        require(follow[-1]['focus_stopped'] == branch['focus_stopped'], 'Stop trace mismatch')
        reasons = json.loads(branch['stop_reasons'])
        require(reasons in (['adaptive'], ['step_cap'], ['probe_horizon']), 'Unknown stop reason')
        if reasons == ['probe_horizon']:
            require(len(steps) == horizon and not branch['focus_stopped'], 'False horizon stop')
        elif reasons == ['adaptive']:
            require(branch['focus_stopped'], 'Adaptive stop without done')
        else:
            require(branch['completed_step'] == report.get('max_denoising_steps', 48), 'False step cap')
        flat = [r for o in trace for r in o['records']]
        with (branch_dir / 'position_probabilities.csv').open(newline='') as stream:
            csv_rows = list(csv.DictReader(stream))
        require(len(csv_rows) == len(flat), 'CSV/JSON record count mismatch')
        for a, b in zip(flat, csv_rows):
            require(set(a) == set(b) and all(b[k] == ('' if v is None else str(v)) for k, v in a.items()),
                    'CSV/JSON data mismatch')
        for observation in trace:
            rows = observation['records']
            require(len(rows) == len(expected_rows) and
                    {(r['position'], r['token_id']) for r in rows} == expected_rows, 'Incomplete window')
            for row in rows:
                require(row['completed_step'] == observation['completed_step'] and
                        row['phase'] == observation['phase'] and
                        row['focus_stopped'] == observation['focus_stopped'], 'Row context mismatch')
                require(np.isfinite(row['probability']) and 0 <= row['probability'] <= 1, 'Bad probability')
            require({s['token_id'] for s in observation['summary']} == set(report['watched_tokens']), 'Missing token summary')
            for summary in observation['summary']:
                token_rows = [r for r in rows if r['token_id'] == summary['token_id']]
                mass = sum(r['probability'] for r in token_rows)
                centroid = sum(r['position'] * r['probability'] for r in token_rows) / mass if mass > 1e-12 else None
                require(np.isclose(mass, summary['local_mass'], rtol=1e-12, atol=1e-14), 'Bad local mass')
                require((centroid is None and summary['local_centroid'] is None) or
                        (centroid is not None and summary['local_centroid'] is not None and
                         np.isclose(centroid, summary['local_centroid'], rtol=1e-12)), 'Bad centroid')
        stem = report['case_id'].replace(':', '-')
        with np.load(branch_dir / (stem + '.npz'), allow_pickle=False) as archive:
            arrays[name] = {k: archive[k] for k in archive.files}
        data = arrays[name]
        require(data['step_indices'].tolist() == list(range(branch['completed_step'])), 'Incomplete canvas0 NPZ')
        require(np.all(data['canvas_indices'] == 0), 'Unexpected multi-canvas trace')
        for observation in trace:
            index = observation['completed_step'] - 1
            for row in observation['records']:
                p = row['position']
                require(int(data['after'][index, p]) == row['canvas_token_id'] and
                        int(data['sampled'][index, p]) == row['sampled_token_id'] and
                        bool(data['accepted'][index, p]) == row['accepted_at_last_step'], 'NPZ/position trace mismatch')
        for filename in (branch['steps_csv'], branch['tokens_csv']):
            require((branch_dir / filename).is_file(), 'Missing recorder CSV')
        if name != 'baseline':
            require(trace[0] == traces['baseline'][0], 'Branches start from different observations')
            edit = report['branches'][name]['edit']
            require(branch['edits'][-1] == edit, 'Edit log mismatch')
            before = {(r['position'], r['token_id']): r['probability'] for r in trace[0]['records']}
            after = {(r['position'], r['token_id']): r['probability'] for r in trace[1]['records']}
            token = report['anchor_token_id']
            s, t = edit['source_position'], edit['target_position']
            require(np.isclose(before[s, token] - after[s, token], edit['actual_source_loss'], atol=1e-14) and
                    np.isclose(after[t, token] - before[t, token], edit['actual_destination_gain'], atol=1e-14),
                    'Actual dose does not match observations')
            require(all(after[k] == v for k, v in before.items() if k[0] not in (s, t)), 'Immediate unedited-row change')
    controls = {}
    for a, b in (('baseline', 'noop'), ('forward', 'repeat_forward')):
        key = a + '_vs_' + b
        controls[key] = dict(
            observed_trace_equal=_without_after(traces[a]) == _without_after(traces[b]),
            exported_focus_npz_equal=set(arrays[a]) == set(arrays[b]) and all(
                np.array_equal(arrays[a][k], arrays[b][k]) for k in arrays[a]),
            recorder_csv_bytes_equal=all((folder/a/branches[a][k]).read_bytes() ==
                (folder/b/branches[b][k]).read_bytes() for k in ('steps_csv', 'tokens_csv')),
            decoded_canvas_equal=branches[a]['decoded_canvas'] == branches[b]['decoded_canvas'])
        require(all(controls[key].values()), 'Saved control mismatch: ' + key)
    require(traces['noop'][1]['records'] == [dict(r, phase='after_edit') for r in traces['noop'][0]['records']],
            'No-op after-edit mismatch')
    require(traces['forward'] == traces['repeat_forward'], 'Repeated edit differs')
    return report, branches, traces, arrays, controls


def moments(rows):
    mass = sum(r['probability'] for r in rows)
    return mass, sum(r['position'] * r['probability'] for r in rows) / mass if mass > 1e-12 else None


def analyze_probe(folder, output, tokenizer=None):
    report, branches, traces, arrays, controls = validate_probe(folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source, target = report['source_position'], report['target_position']
    paired, metrics, canvas_metrics = [], [], []
    baseline = {o['completed_step']: o for o in traces['baseline']}
    for name in ('forward', 'reverse'):
        for obs in traces[name]:
            step, phase = obs['completed_step'], obs['phase']
            if step not in baseline:
                canvas_metrics.append(dict(branch=name, completed_step=step, comparable=False,
                                           reason='baseline stopped; censored'))
                continue
            control = baseline[step]
            by_key = {(r['position'], r['token_id']): r for r in control['records']}
            for row in obs['records']:
                b = by_key[row['position'], row['token_id']]
                paired.append(dict(branch=name, **row, edited_row=row['position'] in (source, target),
                                   baseline_probability=b['probability'], delta=row['probability'] - b['probability']))
            for token in report['watched_tokens']:
                for region in ('all', 'edited', 'unedited'):
                    def inside(row):
                        return row['token_id'] == token and (region == 'all' or
                            (row['position'] in (source, target)) == (region == 'edited'))
                    rows = [r for r in obs['records'] if inside(r)]
                    original = [by_key[r['position'], token] for r in rows]
                    mass, centroid = moments(rows)
                    bm, bc = moments(original)
                    delta = np.array([r['probability'] - b['probability'] for r, b in zip(rows, original)])
                    metrics.append(dict(branch=name, completed_step=step, phase=phase, token_id=token, region=region,
                        mass=mass, baseline_mass=bm, delta_mass=mass-bm, centroid=centroid,
                        baseline_centroid=bc, delta_centroid=centroid-bc if centroid is not None and bc is not None else None,
                        probability_l1=float(np.abs(delta).sum()), probability_max_abs=float(np.abs(delta).max())))
            if phase != 'next_model_prediction':
                continue
            for region in ('whole_canvas', 'edited', 'unedited_window', 'outside_window'):
                length = arrays[name]['after'].shape[1]
                positions = np.arange(length)
                low, high = report['window_inclusive']
                mask = (np.ones(length, bool) if region == 'whole_canvas' else
                        np.isin(positions, [source, target]) if region == 'edited' else
                        ((positions >= low) & (positions <= high) & ~np.isin(positions, [source, target]))
                        if region == 'unedited_window' else (positions < low) | (positions > high))
                row = dict(branch=name, completed_step=step, region=region, comparable=True)
                for field in ('accepted', 'sampled', 'after'):
                    a, b = arrays[name][field][step-1], arrays['baseline'][field][step-1]
                    changed = np.flatnonzero((a != b) & mask).tolist()
                    row[field + '_changed_positions'] = changed
                    row[field + '_changed_count'] = len(changed)
                canvas_metrics.append(row)
    decoded = {}
    for name in BRANCHES:
        decoded[name] = dict(final=branches[name]['decoded_canvas'])
        if tokenizer is not None:
            decoded[name]['by_completed_step'] = {str(int(s)+1): tokenizer.decode(ids)
                for s, ids in zip(arrays[name]['step_indices'], arrays[name]['after'])}
            require(decoded[name]['by_completed_step'][str(branches[name]['completed_step'])] == branches[name]['decoded_canvas'],
                    'Current tokenizer does not reproduce saved decode')
    write_csv(output / 'paired_probabilities.csv', paired)
    write_csv(output / 'token_moments.csv', metrics)
    write_json(output / 'canvas_changes.json', canvas_metrics)
    write_json(output / 'decoded_canvases.json', decoded)
    summary = dict(probe=str(Path(folder).resolve()), report=report, saved_controls=controls,
        validation_scope='Saved focus observations/NPZ/CSV; full GPU-state and restoration checks are recorded claims, not reconstructible snapshots.',
        branches={name: dict(completed_step=b['completed_step'], focus_stopped=b['focus_stopped'],
                    stop_reasons=json.loads(b['stop_reasons']), observed_followups=b['completed_step']-report['base_completed_step'],
                    requested_followups=report['horizon'], decoded_equal_baseline=b['decoded_canvas']==branches['baseline']['decoded_canvas'])
                  for name, b in branches.items()},
        independent_design_units=1, intervention_directions=2, validation_only_branches=2,
        scope='canvas0 only; missing followups censored; repeated steps/branches are not independent samples',
        metrics=metrics, canvas_changes=canvas_metrics,
        input_sha256={str(p.relative_to(folder)): sha256(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()})
    write_json(output / 'summary.json', summary)
    plots(report, traces, output)
    return summary


def plots(report, traces, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tokens = report['watched_tokens']
    labels = {12234: 'triple quote', 2060: 'return', 642: 'opening brace', 107: 'newline'}
    positions = list(range(report['window_inclusive'][0], report['window_inclusive'][1]+1))
    def values(obs, token):
        return [next(r['probability'] for r in obs['records'] if r['position']==p and r['token_id']==token) for p in positions]
    baseline = {o['completed_step']: o for o in traces['baseline']}
    fig, axes = plt.subplots(len(tokens), 3, figsize=(15, 3.0*len(tokens)), squeeze=False)
    for i, token in enumerate(tokens):
        obs = traces['baseline']
        im = axes[i, 0].imshow([values(o, token) for o in obs], aspect='auto', vmin=0, vmax=1, cmap='viridis')
        axes[i, 0].set_title(f'{labels.get(token, token)}: baseline p')
        axes[i, 0].set_yticks(range(len(obs)), [str(o['completed_step']) for o in obs])
        fig.colorbar(im, ax=axes[i, 0], shrink=.7)
        for j, name in enumerate(('forward', 'reverse'), 1):
            obs = [o for o in traces[name] if o['completed_step'] in baseline]
            delta = np.array([np.array(values(o, token))-values(baseline[o['completed_step']], token) for o in obs])
            # Symmetric-log scale keeps small responses visible without concealing the imposed edit.
            from matplotlib.colors import SymLogNorm
            limit = max(float(np.abs(delta).max()), 1e-9)
            im = axes[i, j].imshow(delta, aspect='auto', cmap='RdBu_r',
                norm=SymLogNorm(linthresh=1e-8, vmin=-limit, vmax=limit))
            axes[i, j].set_title(f'{name} − own baseline (symlog)')
            axes[i, j].set_yticks(range(len(obs)), [f"{o['completed_step']} " +
                ('IMPOSED' if o['phase']=='after_edit' else 'before' if o['phase']=='before_edit' else 'response') for o in obs])
            fig.colorbar(im, ax=axes[i, j], shrink=.7)
        for ax in axes[i]:
            ax.set_xticks(range(len(positions)), positions, rotation=90)
            for pos in (report['source_position'], report['target_position']):
                ax.axvspan(positions.index(pos)-.5, positions.index(pos)+.5, facecolor='none', edgecolor='black', lw=1)
            ax.set_xlabel('Canvas position (boxed rows directly edited)')
    fig.suptitle('Watched probabilities; each direction uses this probe’s baseline. Missing steps are censored.')
    fig.tight_layout()
    fig.savefig(output / 'position_heatmaps.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(len(tokens), 2, figsize=(12, 3*len(tokens)), squeeze=False)
    for i, token in enumerate(tokens):
        for name in ('baseline', 'forward', 'reverse'):
            obs = _without_after(traces[name])
            steps = [o['completed_step'] for o in obs]
            summaries = [next(s for s in o['summary'] if s['token_id']==token) for o in obs]
            axes[i, 0].plot(steps, [s['local_centroid'] for s in summaries], 'o-', label=name)
            axes[i, 1].plot(steps, [s['local_mass'] for s in summaries], 'o-', label=name)
        axes[i, 0].set_ylabel(labels.get(token, str(token)))
        axes[i, 0].set_title('Local centroid (positions 36–49)')
        axes[i, 1].set_title('Local probability mass')
        for ax in axes[i]:
            ax.set_xlabel('Completed step; after_edit omitted')
            ax.legend()
            ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'centroids.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from gemma import gm
        tokenizer = gm.text.Gemma4Tokenizer(path=str(args.tokenizer))
    summary = analyze_probe(args.probe, args.output, tokenizer)
    if args.tokenizer:
        write_json(args.output/'decode_identity.json', dict(path=str(args.tokenizer.resolve()), sha256=sha256(args.tokenizer)))
    print(json.dumps(dict(output=str(args.output), controls=summary['saved_controls'], branches=summary['branches']), indent=2))


if __name__ == '__main__':
    main()
