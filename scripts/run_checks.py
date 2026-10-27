"""CPU checks, all writes scoped to the release workspace."""
from pathlib import Path
import datetime
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',help='Checkpoint directory for the optional neighbor-routing test')
parser.add_argument('--output',default=str(CODE/'test-results'))
args=parser.parse_args()
LOGS=Path(args.output).resolve()
assert not LOGS.exists(),'Refusing to overwrite test results'
LOGS.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',
         TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
         HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
skipped=[]
tests=[
 ('syntax',[sys.executable,'-c',"from pathlib import Path; fs=list(Path('.').rglob('*.py')); [compile(p.read_text(encoding='utf-8-sig'),str(p),'exec') for p in fs]; print('SYNTAX_OK',len(fs))"]),
 ('train_help',[sys.executable,'dt_train.py','--help']),
 ('infer_help',[sys.executable,'dt_infer.py','--help']),
 ('ssnc_masks',[sys.executable,'-m','tests.test_ssnc_mask']),
 ('state_tensorization',[sys.executable,'-m','tests.test_state_tensorization_canonical_v2']),
 ('top_p',[sys.executable,'-m','tests.test_top_p_sampling_live']),
 ('routing',[sys.executable,'-m','tests.test_ssnc_inference_routing']),
 ('entrypoint_parity',[sys.executable,'-m','tests.verify_canonical_v2_entrypoints']),
 ('mock_smoke',[sys.executable,'-m','tests.test_smoke']),
]
if args.checkpoint:
 checkpoint=Path(args.checkpoint).resolve()/'dt_custom_modules.pt'
 assert checkpoint.is_file(),checkpoint
 tests.append(('neighbor_routing',[sys.executable,'-m','tests.verify_ssnc_unseen_neighbor_routing',
  '--custom_modules',str(checkpoint),'--output_json',str(LOGS/'neighbor_routing.json')]))
else:
 skipped.append({'name':'neighbor_routing','reason':'Supply --checkpoint PATH to test saved mask routing'})
rows=[]
for name,command in tests:
 start=time.monotonic()
 try:
  result=subprocess.run(command,cwd=CODE,env=env,text=True,capture_output=True,timeout=180)
  output=result.stdout+result.stderr
  rc=result.returncode
 except subprocess.TimeoutExpired as exc:
  output='TIMEOUT '+str(exc); rc=124
 (LOGS/(name+'.log')).write_text(output,encoding='utf-8')
 row={'name':name,'returncode':rc,'seconds':round(time.monotonic()-start,2),'log':str(LOGS/(name+'.log'))}
 rows.append(row)
 print(json.dumps(row),flush=True)
 if rc: print(output[-2800:],flush=True)
versions={}
for name in ['torch','transformers','peft','accelerate','bitsandbytes','numpy','scipy','pandas','scikit-learn','matplotlib','datasets','ortools','tqdm','modelscope','seaborn']:
 try: versions[name]=md.version(name)
 except md.PackageNotFoundError: pass
summary={'python':sys.version,'executable':sys.executable,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'versions':versions,'tests':rows,'skipped':skipped,'all_passed':all(x['returncode']==0 for x in rows)}
(LOGS/'checks.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('CHECK_SUMMARY',json.dumps(summary),flush=True)
sys.exit(0 if summary['all_passed'] else 1)
