"""Resident positional extension. Use --verify to run the existing native gate."""
import json
import shlex
import sys
import time

from exp_code.semantic_intervention.debug_console import Console, main
from .resident_position import move, observe, probe


class PositionalConsole(Console):
    intro = ('SC position debugger: existing commands plus posmove, watch, posprobe. '
             'Type help posprobe for the bounded paired pilot.')

    def _require_engine(self):
        if self.engine is None:
            raise ValueError('Load a case first')

    def do_posmove(self, arg):
        """posmove SOURCE_POSITION TARGET_POSITION TOKEN_ID FRACTION: edit SC only."""
        self._require_engine()
        source, target, token, fraction = shlex.split(arg)
        result = move(self.engine, int(source), int(target), int(token), float(fraction), self.focus)
        print(json.dumps(result, indent=2), flush=True)

    def do_watch(self, arg):
        """watch START END TOKEN_ID [TOKEN_ID ...]: inclusive window; save one observation."""
        self._require_engine()
        start, end, *tokens = map(int, shlex.split(arg))
        result = observe(self.engine, self.focus, start, end, tokens, 'manual_inspection')
        result.update(case_id=self.case_id, seed=self.seed, edits=list(self.engine.edits))
        with (self.output/'position_watch.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result)+'\n')
        print(json.dumps(result, indent=2), flush=True)

    def do_posprobe(self, arg):
        """posprobe SRC DST TOKEN FRACTION HORIZON [OTHER_TOKEN ...]

        Run baseline/no-op/forward/reverse/repeated-forward branches, verify
        no-op and repeatability, export complete watched probabilities at
        every step, then restore the starting state. Native stop stays on.
        Example: posprobe 42 43 12234 0.5 5 2060 642 107
        """
        self._require_engine()
        source, target, token, fraction, horizon, *neighbors = shlex.split(arg)
        folder = self.output/('posprobe-'+str(time.time_ns()))
        report = probe(self.engine, source=int(source), target=int(target),
            token=int(token), fraction=float(fraction), horizon=int(horizon),
            batch=self.focus, case_id=self.case_id, seed=self.seed,
            tokenizer=self.chat.tokenizer, output=folder,
            neighbors=list(map(int, neighbors)))
        print('POSITION_GATE_AND_PROBE', report['status'], folder, flush=True)
        self.do_status('')


if __name__ == '__main__':
    if '--verify' not in sys.argv:
        raise SystemExit('Launch this positional entrypoint with --verify')
    main(console_class=PositionalConsole)
