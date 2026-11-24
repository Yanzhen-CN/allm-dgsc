"""CPU provenance audit of the two historical sessions; never infer drift cause."""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np

from exp_code.baselines.data import digest
from exp_code.common.debug_config import resolve_batch
from .analyze import read_json, sha256, write_csv, write_json
from .run_batch import ROOT, environment


def reconstructed_sources(output):
    """Match the historical digest using git HEAD and the preserved installer.

    These are content reconstructions, not proof of which modules a live
    process had imported or subsequently reloaded.
    """
    paths = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', 'exp_code', 'dg/gemma'], text=True).splitlines()
    head = {p: subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:'+p]) for p in paths if p.endswith('.py')}
    tree = ast.parse((ROOT/'install_direction2_posmove.py').read_text())
    payload = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'PAYLOAD' for t in n.targets))
    extension = dict(head)
    extension.update({p: value.encode() for p, value in payload.items() if p.startswith('exp_code/') and p.endswith('.py')})
    shared = 'exp_code/semantic_intervention/debug_console.py'
    extension[shared] = (ROOT/shared).read_bytes()
    answer = {}
    for label, files in (('earlier', head), ('latest', extension)):
        hashes = {p: hashlib.sha256(value).hexdigest() for p, value in sorted(files.items())}
        write_json(output/(label+'_reconstructed_source_files.json'), hashes)
        with tarfile.open(output/(label+'_reconstructed_source.tar.gz'), 'w:gz') as archive:
            for p, content in files.items():
                info = tarfile.TarInfo(p)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        answer[label] = dict(source_sha256=digest(hashes), files=len(hashes),
            method='git HEAD' if label=='earlier' else 'git HEAD + original installer payload + shared console file',
            runtime_import_identity_verified=False)
    answer['changed_paths'] = [p for p in sorted(head.keys() | extension.keys()) if head.get(p) != extension.get(p)]
    return answer


def score_function_only(text):
    """Repository AST-allowlisted harness in a fresh isolated Python process."""
    program = ('import json,sys; sys.path.insert(0, '+repr(str(ROOT/'scripts'))+'); '
               'from review_filter_operator import function_tests; '
               'print(json.dumps(function_tests(sys.stdin.read())))')
    with tempfile.TemporaryDirectory(prefix='direction2-function-') as directory:
        result = subprocess.run([sys.executable, '-I', '-c', program], input=text, text=True,
            capture_output=True, cwd=directory, timeout=15,
            env={'PATH': os.environ.get('PATH', ''), 'PYTHONIOENCODING': 'utf-8'}, check=True)
    return json.loads(result.stdout)


def audit(earlier, latest, probe, output, function_tests=False):
    output.mkdir(parents=True, exist_ok=False)
    config = read_json(ROOT/'configs/debugger_run.json')
    checkpoint = Path(config['checkpoint'])
    manifest = read_json(checkpoint/'verified-manifest.json')
    prefix = 'checkpoints/diffusiongemma-26B-A4B-it/'
    sizes = []
    for entry in manifest['files']:
        relative = entry['name'][len(prefix):] if entry['name'].startswith(prefix) else 'tokenizer.model'
        sizes.append(dict(path=relative, present=(checkpoint/relative).is_file(),
            size_matches=(checkpoint/relative).stat().st_size == entry['size'] if (checkpoint/relative).is_file() else False))
    source = reconstructed_sources(output)
    sessions = {}
    for label, directory in (('earlier', earlier), ('latest', latest)):
        session = read_json(directory/'session.json')
        gate = read_json(directory/'checkpoint_gate.json')
        with np.load(directory/'diagnostic_native_first8.npz', allow_pickle=False) as a, \
                np.load(directory/'diagnostic_debugger_first8.npz', allow_pickle=False) as b:
            first8 = {key: bool(np.array_equal(a[key], b[key])) for key in a.files}
        with np.load(directory/'sc_carry_diagnostic.npz', allow_pickle=False) as archive:
            carry = {key.removeprefix('native_'): bool(np.array_equal(archive[key], archive['debugger_'+key.removeprefix('native_')]))
                     for key in archive.files if key.startswith('native_') and 'debugger_'+key.removeprefix('native_') in archive.files}
        sessions[label] = dict(path=str(directory), recorded=session,
            source_reconstruction=source[label], declared_source_digest_matches_reconstruction=source[label]['source_sha256']==session['source_sha256'],
            gate=gate, saved_gsm_first8_all_arrays_equal=first8, saved_gsm_sc_carry_equal=carry,
            native_diagnostic=read_json(directory/'native_step_diagnostic.json'),
            commands=[json.loads(line) for line in (directory/'commands.jsonl').read_text().splitlines()],
            file_sha256={str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()},
            missing=['GPU driver/version/UUID and allocation ID', 'environment flags at runtime',
                'dependency versions beyond recorded Python/JAX', 'historically hashed/copied config and rendered prompts',
                'runtime imported-module identities and complete compilation/cache/warmup log',
                'checkpoint payload hashes recomputed in that session', 'MBPP native matched parity gate',
                'two fresh MBPP prefills in one process', 'cross-process full MBPP GPU-state comparison'])
    drift = []
    with (earlier/'mbpp277_baseline_t2/mbpp-277_tokens.csv').open(encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r['step_number'])==2 and int(r['canvas_position']) in (42,43)]
    latest_observation = read_json(probe/'baseline/position_trace.json')[0]
    for row in rows:
        position = int(row['canvas_position'])
        ids, probs = json.loads(row['top_token_ids']), json.loads(row['top_probabilities'])
        for token in (12234,2060,642,107):
            observation = next(r for r in latest_observation['records'] if r['position']==position and r['token_id']==token)
            old = probs[ids.index(token)] if token in ids else None
            drift.append(dict(position=position, token_id=token, earlier_p=old, latest_p=observation['probability'],
                delta=observation['probability']-old if old is not None else None,
                earlier_probability_method='saved float32 full-softmax probability of top16 entry; not renormalized',
                latest_probability_method='float64 full-vocabulary softmax of stored logits',
                earlier_canvas=int(row['output_token_id']), latest_canvas=observation['canvas_token_id'],
                earlier_accepted=row['accepted']=='True', latest_accepted=observation['accepted_at_last_step']))
    write_csv(output/'baseline_drift.csv', drift)
    batch, focus, seed = resolve_batch(config, 'mbpp:277')
    result = dict(audit_utc=datetime.now(timezone.utc).isoformat(), sessions=sessions,
        reconstructed_source_changes=source['changed_paths'], baseline_drift=drift,
        drift_cause='Not established. Different declared source identities do not establish a causal explanation.',
        current_environment=environment(), current_run_config=dict(sha256=sha256(ROOT/'configs/debugger_run.json'), contents=config),
        batch_identity=dict(batch=batch, focus=focus, seed=seed,
            basis='Current config resolver and user-provided original identity; historical session only stores config path, focus/seed/case and commands, not the four rendered prompts.'),
        checkpoint=dict(path=str(checkpoint), manifest_file_sha256=sha256(checkpoint/'verified-manifest.json'),
            declared_manifest_digest=manifest['sha256'], manifest_self_consistent=digest(manifest['files'])==manifest['sha256'],
            sizes=sizes, tokenizer_sha256_now=sha256(checkpoint/'tokenizer.model'),
            tokenizer_matches_manifest_now=sha256(checkpoint/'tokenizer.model')==next(e['sha256'] for e in manifest['files'] if 'tokenizers/' in e['name']),
            payload_content_verified_now=False,
            note='Payload sizes and manifest integrity checked now; weight bytes not rehashed by CPU historical audit. Batch runner hashes all payloads before loading. No historical runtime weight attestation.'),
        earlier_export_stop_warning='Both t2 and t3 exports have partial=true and stop_reasons=[step_cap]. Their step CSVs do not report cap/adaptive stop; these are unfinished manual breakpoints, not native stops.',
        previous_day_archive='results/interim_sc_case_study/provenance describes a different Nov 19 session; not substituted for missing metadata of these sessions.')
    write_json(output/'provenance.json', result)
    if function_tests:
        scores = {name: score_function_only(read_json(probe/name/'branch.json')['decoded_canvas'])
                  for name in ('baseline','noop','forward','reverse','repeat_forward')}
        write_json(output/'function_tests.json', dict(scope='Extracted complete dict_filter function only; not full-response benchmark accuracy.',
            harness='scripts/review_filter_operator.py:function_tests', harness_sha256=sha256(ROOT/'scripts/review_filter_operator.py'),
            execution='Fresh python -I child, stripped environment, temporary cwd, AST allowlist; no unrelated generated tests/text executed.',
            independent_function_variants=len({read_json(probe/name/'branch.json')['decoded_canvas'] for name in scores}),
            scores=scores))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--earlier', type=Path, default=Path('outputs/position-session-20261120-170358'))
    parser.add_argument('--latest', type=Path, default=Path('outputs/position-pilot-20261120-175225'))
    parser.add_argument('--probe', type=Path, default=Path('outputs/position-pilot-20261120-175225/posprobe-1795216132241328508'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--function-tests', action='store_true')
    args = parser.parse_args()
    result = audit(args.earlier, args.latest, args.probe, args.output, args.function_tests)
    print(json.dumps(dict(output=str(args.output), source_matches={k: v['declared_source_digest_matches_reconstruction']
        for k,v in result['sessions'].items()}, drift_cause=result['drift_cause']), indent=2))


if __name__ == '__main__':
    main()
