import json,hashlib
from pathlib import Path
w=Path.cwd();base=w/'discussion/ablation_study'
pointer=json.loads((base/'active_run.json').read_text())
r=base/'results'/pointer['run_name']
identity=r/'experiment_identity.json';old=json.loads(identity.read_text())
new=json.loads((base/'checks/preflight_report.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=old['config']
signature=dict(config=cfg,sources=old['preflight']['sources'],reference_weight=sha(cfg['reference_checkpoint']),pure_weight=sha(cfg['pure_checkpoint']),dataset_stat=[Path(cfg['dataset']).stat().st_size,Path(cfg['dataset']).stat().st_mtime_ns],label_cache=sha(cfg['label_cache']))
hash_signature=lambda s:hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest()
assert hash_signature(signature)==old['fingerprint'],'Original configuration/data/weight identity no longer matches'
changes={k:dict(before=v,after=new['sources'].get(k)) for k,v in old['preflight']['sources'].items() if new['sources'].get(k)!=v}
assert set(new['sources'])==set(signature['sources'])
assert set(changes)=={str(Path('discussion/ablation_study')/f) for f in ['ablation_training.py','ablation_runner.py']},changes
before=json.loads((base/'checks/gradient_repair_before.json').read_text())
assert all(changes[str(Path('discussion/ablation_study')/k)]['before']==v for k,v in before.items())
signature['sources']=new['sources']
assert hash_signature(signature)==new['fingerprint']
completed=[]
for state_file in r.glob('*/state.json'):
 state=json.loads(state_file.read_text())
 if state['status']=='complete':
  assert state['fingerprint']==old['fingerprint']
  assert all(sha(state_file.parent/p)==h for p,h in state['files'].items())
  completed.append(state_file.parent.name)
checkpoint=r/'Single_azimuth/resume_checkpoint.pth'
import torch
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert state['epoch']==49 and state['fingerprint']==old['fingerprint']
assert all(torch.isfinite(v).all() for v in state['model_state_dict'].values())
receipt=dict(training_fingerprint=old['fingerprint'],runtime_fingerprint=new['fingerprint'],identity_sha256=sha(identity),source_changes=changes,verified_completed_models=completed,resume_checkpoint_sha256=sha(checkpoint),resume_epoch=49,configuration_data_and_reference_weights_unchanged=True,formal_training_started=False,scope='Numerical gradient guard and one same-batch retry without TF32; existing checkpoint identities preserved. Not evidence of original failure cause.')
(r/'gradient_repair_compatibility.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
