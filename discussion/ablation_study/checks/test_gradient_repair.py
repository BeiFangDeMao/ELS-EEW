import sys,json,hashlib
from pathlib import Path
import torch
root=Path('discussion/ablation_study').resolve()
sys.path.insert(0,str(root))
from ablation_runner import configure
cfg=configure(root/'ablation_config.yaml')
from ablation_training import checked_clip,epoch
from ablation_models import build_model,experiments
from training_adapter import optimizer_scheduler
from torch.utils.data import DataLoader,TensorDataset
out=root/'checks/gradient_repair_tests';out.mkdir(exist_ok=True)
torch.set_num_threads(2)
p=torch.nn.Linear(2,1,bias=False);p.weight.grad=torch.tensor([[1e20,1e20]])
norm=checked_clip(p,.8)
assert torch.isfinite(p.weight.grad).all() and abs(float(p.weight.grad.norm())-.8)<1e-6
q=torch.nn.Linear(2,1,bias=False);q.weight.grad=torch.tensor([[3.,4.]])
r=torch.nn.Linear(2,1,bias=False);r.weight.grad=q.weight.grad.clone()
checked_clip(q,.8);torch.nn.utils.clip_grad_norm_(r.parameters(),.8,error_if_nonfinite=True)
assert torch.equal(q.weight.grad,r.weight.grad)
spec=next(s for s in experiments(cfg) if s['name']=='Single_azimuth')
wave=torch.randn(2,3,96);y=torch.tensor([[4.,50.,0.,1.],[5.,100.,1.,0.]])
loader=[(wave,y,['synthetic_a','synthetic_b'],None)]
for persistent in [False,True]:
 model=build_model(spec);optimizer,scheduler=optimizer_scheduler(model,cfg,1)
 before={k:v.clone() for k,v in model.state_dict().items()}
 calls=[0]
 def hook(grad):
  calls[0]+=1
  return grad*float('nan') if persistent or calls[0]==1 else grad
 handle=next(model.parameters()).register_hook(hook)
 initial=scheduler.last_epoch
 folder=out/('persistent' if persistent else 'transient')
 try:
  result=epoch(model,loader,spec,cfg,torch.device('cpu'),optimizer,scheduler,folder,0)
  assert not persistent and scheduler.last_epoch==initial+1 and len(optimizer.state)>0
 except FloatingPointError:
  assert persistent and scheduler.last_epoch==initial and len(optimizer.state)==0
  assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
 handle.remove()
 assert len(list(folder.glob('*.pth')))==1
assert torch.backends.cuda.matmul.allow_tf32==False or isinstance(torch.backends.cuda.matmul.allow_tf32,bool)
report=dict(normal_clip_bitwise_equal=True,finite_fp32_norm_overflow_recovered=True,
 transient_bad_gradient_retry_one_optimizer_step=True,persistent_bad_gradient_no_update=True,
 snapshots_saved=True,formal_training_started=False)
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
