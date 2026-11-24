"""Resident model, one-canvas debugger. `until 5` pauses AFTER step 5.

First version deliberately keeps breakpoints inside one canvas; it does not
claim that CSVs or a new process can restore GPU-resident snapshots.
"""
import argparse
import cmd
import gc
import json
from pathlib import Path
import shlex
import signal
import time
from exp_code.common.debug_config import resolve_batch


def prepare_canvas(chat,prompts,seed):
  import jax
  from gemma.gm.text import _prefill,_sampler
  from exp_code.common.canvas_debugger import CanvasDebugger
  inputs=chat._get_inputs(prompt=prompts,images=None,add_bos=True,
                          has_batch_dim=True,sharding=None)
  initial=_prefill.prefill(model=chat.model,params=chat.params,input=inputs,
      last_state=None,cache_length=chat.cache_length,pad_length=chat.pad_length,
      rng=_sampler._normalize_rng(seed),sharding=None,max_out_length=chat.max_out_length)
  _,canvas_rng=jax.random.split(initial.rng)
  loop=chat._initialize_sampler_loop(chat.sampling)
  engine=CanvasDebugger(sampler=loop,params=chat.params,rng=canvas_rng,
      batch_size=len(prompts),cache=initial.cache,full_attention_mask=initial.full_attention_mask)
  return engine,initial


def verify_checkpoint(chat,previous,output):
  """Actual weights: native parity, exact no-op restore and repeated SC fork."""
  import dialog
  import jax
  import numpy as np
  from gemma.diffusion import _sampler
  cases=previous['plans']['gsm8k']['cases'][:4]
  focus=next(i for i,c in enumerate(cases) if c['id']=='gsm8k:480')
  prompts=[dialog.Conversation(dialog.User(c['prompt'])).as_text(format=chat.tokenizer.FORMAT) for c in cases]
  seed=previous['seed']
  import jax.numpy as jnp
  eager_noise=1.-jnp.arange(49)/48
  compiled_noise=jax.jit(lambda:1.-jnp.arange(49)/48)()
  print('NOISE_SCHEDULE_FLOAT32_DIAGNOSTIC',
      np.flatnonzero(np.asarray(eager_noise)!=np.asarray(compiled_noise)).tolist(),flush=True)
  print('DEBUGGER_NATIVE_GATE_START',flush=True)
  result=chat.sample(prompts,rng=seed,max_new_tokens=256,return_state=True)
  jax.block_until_ready(result.tokens)
  native=np.array(result.tokens,copy=True)[:,:256]
  del result;gc.collect()
  import dataclasses
  from exp_code.common.acceptance_trace import AcceptanceRecorder,TracedPredictions
  from exp_code.common.carry_diagnostic import CarryChat
  from exp_code.common.sc_sampler import _init_fields
  native_sc={}
  def capture_sc(step,sc):native_sc[int(step)]=np.array(sc,copy=True)
  native_trace=AcceptanceRecorder(4,chat._initialize_sampler_loop(chat.sampling).end_tokens,
      progress_label='official-native-diagnostic')
  traced_fields=_init_fields(chat)
  traced_fields['sample_from_predictions']=TracedPredictions(
      callback=native_trace.record,early_stop_fn=chat.early_stop_fn,
      text_vocab_size=chat.tokenizer.vocab_size,top_k=16)
  traced_chat=CarryChat(**traced_fields,sc_callback=capture_sc)
  result=traced_chat.sample(prompts,rng=seed,max_new_tokens=256,return_state=True)
  jax.block_until_ready(result.tokens);jax.effects_barrier()
  np.testing.assert_array_equal(native,np.asarray(result.tokens)[:,:256])
  del result,traced_chat;gc.collect()
  engine,initial=prepare_canvas(chat,prompts,seed)
  engine.recorder.progress_label='debugger-diagnostic'
  engine.save('start')
  debugger_sc={0:np.array(engine.state.sc_embeddings,copy=True)}
  for _ in range(5):
    engine.step()
    if engine.state.step<=2:debugger_sc[engine.state.step]=np.array(engine.state.sc_embeddings,copy=True)
  engine.save('base')
  sc_difference={step:dict(unequal=int(np.count_nonzero(native_sc[step]!=debugger_sc[step])),
      max_abs=float(np.max(np.abs(native_sc[step].astype(np.float32)-debugger_sc[step].astype(np.float32)))))
      for step in native_sc if step in debugger_sc}
  print('SC_CARRY_FIRST_DIFFERENCE',sc_difference,flush=True)
  np.savez_compressed(output/'sc_carry_diagnostic.npz',**{
      **{'native_'+str(k):v for k,v in native_sc.items()},
      **{'debugger_'+str(k):v for k,v in debugger_sc.items()}})
  canvas_at_break=np.array(engine.state.canvas)
  rng_at_break=np.array(jax.random.key_data(engine.state.rng))
  engine.run();raw=np.array(engine.state.canvas)
  truncated,_=_sampler._truncate_canvas_at_stop_tokens(engine.state.canvas,
      end_tokens=engine.sampler.end_tokens,canvas_length=256,done=initial.done)
  differences=[]
  for a,b in zip(native_trace.events,engine.recorder.events):
    counts={key:int(np.count_nonzero(a[key]!=b[key])) for key in (
        'before','sampled','accepted','after','top_ids','top_logits','stop','current','target')}
    if any(counts.values()):
      differences.append(dict(step=a['step']+1,counts=counts,
          entropy_max_delta=float(np.max(np.abs(a['entropy']-b['entropy']))),
          accepted_native=a['accepted'].sum(-1).tolist(),
          accepted_debugger=b['accepted'].sum(-1).tolist()))
  diagnostic=dict(native_steps=len(native_trace.events),debugger_steps=len(engine.recorder.events),
      differences=differences)
  (output/'native_step_diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n',encoding='utf-8')
  for label,events in [('native',native_trace.events),('debugger',engine.recorder.events)]:
    np.savez_compressed(output/f'diagnostic_{label}_first8.npz',**{
        key:np.stack([event[key] for event in events[:8]]) for key in
        ('before','sampled','accepted','after','entropy','top_ids','top_probs','top_logits','stop')})
  print('NATIVE_STEP_FIRST_DIFFERENCE',differences[:1],flush=True)
  np.testing.assert_array_equal(native,truncated)
  engine.restore('base');np.testing.assert_array_equal(engine.run().canvas,raw)
  branches=[]
  for _ in range(2):
    engine.restore('base')
    engine.swap(position=104,source=236778,target=236812,batch=focus)
    np.testing.assert_array_equal(engine.state.canvas,canvas_at_break)
    np.testing.assert_array_equal(jax.random.key_data(engine.state.rng),rng_at_break)
    branches.append(np.array(engine.run().canvas))
  np.testing.assert_array_equal(branches[0],branches[1])
  engine.restore('base')
  engine.reference_canvas_tokens=native.copy()
  output.mkdir(parents=True,exist_ok=True)
  (output/'checkpoint_gate.json').write_text(json.dumps(dict(passed=True,
      native_first_canvas_exact=True,restore_noop_exact=True,repeat_swap_exact=True,
      scope='actual checkpoint, original four-case batch; snapshot after step5',
      case_id='gsm8k:480',focus=focus,seed=seed),indent=2)+'\n',encoding='utf-8')
  print('DEBUGGER_CHECKPOINT_GATE_PASSED; resident model ready at base breakpoint',flush=True)
  return engine,initial,focus,seed


def resident_checkpoint_gate(chat,previous,output):
  """A failed gate pauses safely instead of discarding the loaded weights."""
  import importlib
  import traceback
  verifier=verify_checkpoint
  attempt=0
  while True:
    attempt+=1
    try:return verifier(chat,previous,output)
    except Exception:
      detail=traceback.format_exc()
      output.mkdir(parents=True,exist_ok=True)
      (output/f'gate_failure_{attempt}.json').write_text(json.dumps(dict(
          passed=False,attempt=attempt,error=detail,model_retained=True),indent=2)+'\n',encoding='utf-8')
      print(detail,flush=True)
      print('GATE FAILED. Weights remain resident. Experiment commands are blocked until the gate passes.',flush=True)
    gc.collect()
    while True:
      action=input('gate-failed> retry / quit: ').strip().lower()
      if action=='quit':raise SystemExit('Explicit quit; releasing GPU')
      if action=='retry':break
      print('Use retry after a reviewed code fix, or quit.',flush=True)
    # Controlled reload only at this idle gate boundary; never while a model
    # step is running. chat/params stay the same objects and are not restored.
    importlib.reload(importlib.import_module('exp_code.common.canvas_debugger'))
    revised=importlib.reload(importlib.import_module('exp_code.semantic_intervention.debug_console'))
    verifier=revised.verify_checkpoint


class Console(cmd.Cmd):
  intro='SC debugger: load CASE; break N; continue; save NAME; inspect POS; swap POS SOURCE_ID TARGET_ID; step; restore NAME; export NAME; quit. All steps are one-based completed-step counts.'
  prompt='sc-debug> '
  def __init__(self,chat,previous,output,idle_seconds=900):
    super().__init__();self.chat=chat;self.previous=previous;self.output=output
    self.engine=None;self.focus=0;self.case_id=None;self.seed=None
    self.break_after=None
    self.idle_seconds=idle_seconds;self.output.mkdir(parents=True,exist_ok=True)

  def precmd(self,line):
    signal.alarm(0)
    with (self.output/'commands.jsonl').open('a',encoding='utf-8') as stream:
      stream.write(json.dumps(dict(time=time.time(),case_id=self.case_id,command=line))+'\n')
    return line

  def postcmd(self,stop,line):
    signal.alarm(self.idle_seconds);return stop

  def onecmd(self,line):
    if self.engine is None and line.split(' ',1)[0] in (
        'until','step','continue','save','restore','drop','inspect','swap','mix','canvas','zero','region','export','break','text','explore'):
      print('ERROR Load a case first',flush=True);return False
    try:return super().onecmd(line)
    except (Exception,KeyboardInterrupt):
      self.recover_command_error(line)
      return False

  def recover_command_error(self,line):
    import traceback
    detail=traceback.format_exc()
    restored=None
    if self.engine is not None and self.engine.snapshots:
      restored=next(reversed(self.engine.snapshots))
      self.engine.restore(restored)
    record=dict(time=time.time(),command=line,error=detail,
        model_retained=True,restored_snapshot=restored)
    try:
      with (self.output/'command_errors.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record)+'\n')
    except OSError as error:
      print('Could not persist error log:',str(error),flush=True)
    print(detail,flush=True)
    print('COMMAND FAILED; weights retained; restored snapshot:',restored,flush=True)

  def emptyline(self):pass

  def do_load(self,arg):
    import dialog
    case_id=arg.strip()
    # Original batch4 shape and RNG, including the other batch members.
    batch,focus,self.seed=resolve_batch(self.previous,case_id)
    prompts=[dialog.Conversation(dialog.User(c['prompt'])).as_text(format=self.chat.tokenizer.FORMAT) for c in batch]
    self.engine=None;self.initial=None;self.break_after=None;gc.collect()
    self.engine,self.initial=prepare_canvas(self.chat,prompts,self.seed)
    self.focus=focus;self.case_id=case_id;self.engine.save('start')
    print('PREFILLED',case_id,'focus',self.focus,'seed',self.seed,'canvas0 only',flush=True)

  def do_until(self,arg):
    self.engine.until(int(arg));self.do_status('')

  def do_step(self,arg):
    for _ in range(int(arg or 1)):
      if self.engine.finished:break
      self.engine.step()
    self.do_status('')

  def do_continue(self,arg):
    if self.break_after is not None and self.engine.state.step<self.break_after:
      self.engine.until(self.break_after);self.do_status('');return
    self.engine.run();self.do_status('')
    from gemma.diffusion import _sampler
    canvas,_=_sampler._truncate_canvas_at_stop_tokens(self.engine.state.canvas,
        end_tokens=self.engine.sampler.end_tokens,canvas_length=self.engine.sampler.canvas_length,
        done=self.initial.done)
    print(self.chat.tokenizer.decode(canvas[self.focus]),flush=True)

  def do_break(self,arg):
    target=int(arg)
    if not self.engine.state.step<target<=self.engine.sampler.max_denoising_steps:
      raise ValueError('Breakpoint must be a future step within this canvas')
    self.break_after=target;print('BREAK AFTER STEP',target)

  def do_save(self,arg):self.engine.save(arg.strip());print('SAVED',arg.strip())
  def do_restore(self,arg):self.engine.restore(arg.strip());self.do_status('')
  def do_drop(self,arg):del self.engine.snapshots[arg.strip()]

  def do_inspect(self,arg):
    row=self.engine.inspect(int(arg),batch=self.focus)
    row['tokens']=[self.chat.tokenizer.decode([i]) for i in row['token_ids']]
    print(json.dumps(row,ensure_ascii=False,indent=2),flush=True)

  def do_swap(self,arg):
    position,source,target=map(int,shlex.split(arg))
    print(json.dumps(self.engine.swap(position=position,source=source,target=target,batch=self.focus)),flush=True)

  def do_mix(self,arg):
    fields=shlex.split(arg);position=int(fields[0]);alpha=float(fields[1])
    pairs=[value.split(':') for value in fields[2:]]
    weights={int(token):float(weight) for token,weight in pairs}
    if len(weights)!=len(pairs):raise ValueError('Duplicate distribution token ID')
    print(json.dumps(self.engine.mix(position=position,alpha=alpha,weights=weights,batch=self.focus)),flush=True)

  def do_canvas(self,arg):
    position,token=map(int,shlex.split(arg))
    print(json.dumps(self.engine.change_canvas(position=position,token=token,batch=self.focus)),flush=True)

  def do_zero(self,arg):
    print(json.dumps(self.engine.zero_sc(position=int(arg),batch=self.focus)),flush=True)

  def do_region(self,arg):
    fields=shlex.split(arg)
    print(json.dumps(self.engine.region(start=int(fields[0]),alpha=float(fields[1]),
        tokens=list(map(int,fields[2:])),batch=self.focus)),flush=True)

  def do_text(self,arg):
    print(self.chat.tokenizer.decode(self.engine.state.canvas[self.focus]),flush=True)

  def do_explore(self,arg):
    from exp_code.semantic_intervention.iterative_probe import explore_committed
    if arg.strip():position,source,target=map(int,shlex.split(arg))
    elif self.case_id=='gsm8k:480':position,source,target=104,236778,236812
    else:raise ValueError('Specify POSITION SOURCE_ID TARGET_ID for this case')
    report=explore_committed(self.engine,focus=self.focus,position=position,
        source=source,target=target,case_id=self.case_id,seed=self.seed,
        tokenizer=self.chat.tokenizer,output=self.output/f'exploration-{time.time_ns()}')
    print(json.dumps(report),flush=True)

  def do_status(self,arg):
    if self.engine is None:print('Load a case first');return
    s=self.engine.state
    print('CASE',self.case_id,'completed',s.step,'next',s.step+1,
          'canvas_stopped',self.engine.finished,'focus_stopped',bool(s.done[self.focus]),
          'snapshots',list(self.engine.snapshots),flush=True)

  def do_export(self,arg):
    name=arg.strip()
    if not name or Path(name).name!=name or name in ('.','..'):raise ValueError('Use a simple branch name')
    folder=self.output/name
    if folder.exists():raise ValueError('Branch output exists; choose another name')
    folder.mkdir()
    summary=self.engine.recorder.export(folder,self.focus,self.case_id,self.seed,self.chat.tokenizer)
    summary.update(edits=self.engine.edits,partial=not self.engine.finished,
        scope='canvas0 only',decoded_canvas=self.chat.tokenizer.decode(self.engine.state.canvas[self.focus]))
    (folder/'branch.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('EXPORTED',folder,flush=True)

  def do_quit(self,arg):return True
  def do_EOF(self,arg):print();return True


def main(console_class=Console):
  import jax
  import sys
  from exp_code.baselines.model_loading import load_model
  from exp_code.baselines.run import source_identity
  from exp_code.common.sc_sampler import SCSampler
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--run-config',type=Path,
      help='Run plan JSON; defaults to existing benchmark run.json, otherwise configs/debugger_run.json')
  parser.add_argument('--checkpoint',type=Path,help='Override the local checkpoint directory')
  parser.add_argument('--output',type=Path,default=Path('outputs/sc-debug-session'))
  parser.add_argument('--idle-seconds',type=int,default=0,
      help='Optional idle timeout; 0 retains the process for the allocated research block')
  parser.add_argument('--verify',action='store_true',help='Require a real checkpoint restore/fork gate before the console')
  parser.add_argument('--explore',action='store_true',help='After verification, run a bounded committed-site causal probe before the console')
  args=parser.parse_args()
  if args.explore and not args.verify:parser.error('--explore requires --verify')
  run_config=args.run_config
  if run_config is None:
    legacy=Path('outputs/sc-benchmark64-deferred-v2/run.json')
    run_config=legacy if legacy.exists() else Path('configs/debugger_run.json')
  previous=json.loads(run_config.read_text(encoding='utf-8'))
  if args.checkpoint is not None:previous['checkpoint']=str(args.checkpoint)
  args.output.mkdir(parents=True,exist_ok=True)
  if (args.output/'session.json').exists():raise ValueError('Session output already exists; use a new directory')
  (args.output/'session.json').write_text(json.dumps(dict(checkpoint_sha256=previous['checkpoint_sha256'],
      source_sha256=source_identity(),device_kind=[d.device_kind for d in jax.devices()],
      jax=jax.__version__,python=sys.version,scope='canvas0, original batch4 and RNG',
      idle_seconds=args.idle_seconds,run_config=str(run_config),checkpoint=previous['checkpoint'],
      snapshots='GPU memory only; expire with process'),indent=2)+'\n',encoding='utf-8')
  print('SINGLE_WEIGHT_LOAD_START',flush=True)
  model,params,tokenizer,_=load_model(Path(previous['checkpoint']),1)
  chat=SCSampler(model=model,params=params,tokenizer=tokenizer,cache_length=previous['cache_length'],
      max_out_length=previous['max_new_tokens'],canvas_length=256,max_denoising_steps=48)
  console=console_class(chat,previous,args.output,args.idle_seconds)
  if args.verify:
    console.engine,console.initial,console.focus,console.seed=resident_checkpoint_gate(chat,previous,args.output)
    console.case_id='gsm8k:480'
  if args.explore:console.onecmd('explore')
  def idle_exit(signum,frame):raise SystemExit('Idle timeout; releasing GPU. In-memory breakpoints expire.')
  signal.signal(signal.SIGALRM,idle_exit);signal.alarm(args.idle_seconds)
  while True:
    try:
      console.cmdloop()
      break  # Only an explicit quit/EOF ends the resident console normally.
    except (Exception,KeyboardInterrupt):
      console.recover_command_error('console input loop')


if __name__=='__main__':main()
