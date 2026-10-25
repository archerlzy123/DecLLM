"""Portable entrypoints around unchanged paper train/infer implementations."""
from pathlib import Path
import argparse
import datetime
import json
import os
import subprocess
import sys

CODE=Path(__file__).resolve().parents[1]
WORK=CODE
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('action',choices=['train','infer'])
p.add_argument('--config',default=str(CODE/'configs/paths.json'))
p.add_argument('--smoke',action='store_true')
p.add_argument('--gpu',default=None)
p.add_argument('--name',default=None)
p.add_argument('--checkpoint',default=None)
p.add_argument('--jobs',type=int,default=10)
p.add_argument('--machines',type=int,default=5)
p.add_argument('--rollouts',type=int,default=1)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
config_path=Path(a.config)
if not config_path.is_file():
    p.error('Copy configs/paths.example.json to configs/paths.json and set your local resource paths, or use --config PATH')
cfg=json.loads(config_path.read_text(encoding='utf-8'))
gpu=a.gpu if a.gpu is not None else str(cfg.get('gpu','0'))
name=a.name or (a.action+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
assert name and Path(name).name==name and name not in {'.','..'}
out=WORK/'runs'/name
assert not out.exists(),'Refusing to overwrite existing output '+str(out)
base=Path(cfg['base_model']).expanduser().resolve()
assert base.is_dir(),base
python=cfg.get('python') or sys.executable

def setarg(argv,key,value):
    if key in argv:
        idx=argv.index(key)
        argv[idx+1]=str(value)
    else: argv.extend([key,str(value)])

def drop(argv,key):
    if key in argv:
        idx=argv.index(key); del argv[idx:idx+2]

if a.action=='train':
    argv=json.loads((CODE/'configs/training_ssnc_v1.json').read_text(encoding='utf-8'))
    setarg(argv,'--model_name',base)
    dataset=Path(cfg.get('smoke_trajectories') or CODE/'data/smoke_trajectories.jsonl') if a.smoke else Path(cfg['trajectories']).expanduser().resolve()
    assert dataset.is_file(),dataset
    setarg(argv,'--trajectories_path',dataset)
    setarg(argv,'--output_dir',out)
    if a.smoke:
        for key in ['--rollout_panel_jsonl','--rollout_panel_score_weights']: drop(argv,key)
        for key,value in {'--max_length':16,'--num_train_epochs':1,
             '--gradient_accumulation_steps':1,'--max_optimizer_steps':2,
             '--eval_steps':0,'--save_step':999999,'--logging_steps':1,
             '--warmup_ratio':0,'--scale_param_mask_warmup_steps':1,
             '--scale_param_mask_update_interval':1}.items(): setarg(argv,key,value)
    else:
        panel=Path(cfg.get('rollout_panel',''))
        assert panel.is_file(),'Set rollout_panel to the paper fixed validation panel before full training'
        setarg(argv,'--rollout_panel_jsonl',panel)
    command=[python,str(CODE/'dt_train.py')]+argv
else:
    checkpoint=Path(a.checkpoint or cfg['checkpoint']).expanduser().resolve()
    for fn in ['adapter_config.json','dt_custom_modules.pt']:
        assert (checkpoint/fn).is_file(),checkpoint/fn
    command=[python,str(CODE/'dt_infer.py'),
        '--model_dir',str(checkpoint),'--base_model',str(base),'--model_source','huggingface',
        '--n_jobs',str(a.jobs),'--n_machines',str(a.machines),
        '--model_n_jobs','20','--model_n_machines','20',
        '--state_encoder_type','graph','--use_candidate_action_head',
        '--use_scale_adapter_expert','--use_scale_embedding','--graph_disable_queue_pos',
        '--use_scale_param_mask','--mask_gnn_encoder','--mask_state_projection',
        '--mask_lora','--mask_candidate_head','--mask_scale_adapter',
        '--state_tensorization_mode','canonical_v2','--unseen_scale_mask_routing','all_one',
        '--dtype','bfloat16','--max_length','128','--max_ep_len','512',
        '--max_steps',str(a.jobs*a.machines+20),'--n_instances','1',
        '--n_rollouts_per_instance',str(a.rollouts),'--rtg_mode','normalized_budget',
        '--target_return','0.8','--seed','62','--no-visualize','--no-save-gantt',
        '--temperature','0.0' if a.rollouts==1 else '1.0',
        '--top_k','1' if a.rollouts==1 else '0','--top_p','0.0' if a.rollouts==1 else '0.7',
        '--save_selected_jsonl',str(out/'selected_rollouts.jsonl')]
record={'action':a.action,'smoke':a.smoke,'gpu':gpu,'command':command,'cwd':str(out)}
print(json.dumps(record,ensure_ascii=False),flush=True)
if a.dry_run: sys.exit(0)
free=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>=30000,f'GPU {gpu} has only {free} MiB free; refusing to start'
out.mkdir(parents=True,exist_ok=False)
(out/'command.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',JSSP_GRAPH_EDGE_CHUNK_SIZE='65536',
    PYTHONUNBUFFERED='1',TORCHDYNAMO_DISABLE='1',TORCHINDUCTOR_COMPILE_THREADS='1',
    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
with (out/'run.log').open('w',encoding='utf-8') as log:
    child=subprocess.Popen(command,cwd=out,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in child.stdout:
        log.write(line); log.flush(); print(line,end='',flush=True)
    rc=child.wait()
(out/'exit.json').write_text(json.dumps({'returncode':rc}),encoding='utf-8')
sys.exit(rc)
