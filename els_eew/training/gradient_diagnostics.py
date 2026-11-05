"""Save non-finite-gradient evidence without updating parameters."""
import json
import logging
import math
from pathlib import Path
import torch


def gradient_report(model):
    rows = []
    square_sum = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().cpu().double()
        finite = torch.isfinite(grad)
        good = grad[finite]
        norm = float(torch.linalg.vector_norm(grad))
        rows.append(dict(parameter=name, nan=int(torch.isnan(grad).sum()),
                         inf=int(torch.isinf(grad).sum()),
                         abs_max=float(good.abs().max()) if good.numel() else None,
                         norm_fp64=norm))
        square_sum += norm*norm
    total = math.sqrt(square_sum)
    return dict(total_norm_fp64=total,
                all_gradients_finite=all(r["nan"]==0 and r["inf"]==0 for r in rows),
                parameters=rows)


def save_failure(model, optimizer, scheduler, wave, labels, keys, epoch, batch,
                 directory, before_buffers, rng_cpu, rng_cuda, model_id):
    folder = Path(directory)/f"gradient_failure_epoch{epoch:03d}_batch{batch:04d}"
    folder.mkdir(parents=True, exist_ok=False)
    report = gradient_report(model)
    report.update(epoch=epoch,batch_zero_based=batch,model_id=model_id,
                  learning_rates=[g["lr"] for g in optimizer.param_groups],
                  sequence_length=wave.shape[-1], torch_version=str(torch.__version__),
                  cuda_version=torch.version.cuda,
                  device=str(wave.device))
    (folder/"gradient_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    # Parameters have not been stepped. Restore pre-forward BN buffers for replay.
    state = {name: tensor.detach().cpu().clone() for name,tensor in model.state_dict().items()}
    for name, tensor in before_buffers.items():
        state[name] = tensor.cpu()
    torch.save(dict(model_id=model_id,model_state_dict=state,
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    wave=wave.detach().cpu(),labels=labels.detach().cpu(),keys=list(keys),
                    epoch=epoch,batch=batch,rng_cpu=rng_cpu,rng_cuda=rng_cuda,
                    sequence_length=wave.shape[-1]),folder/"replay_batch.pth")
    bad = [r["parameter"] for r in report["parameters"] if r["nan"] or r["inf"]]
    logging.error("Gradient evidence saved: %s; FP64 norm=%s; non-finite parameters=%s",
                  folder,report["total_norm_fp64"],bad)
    return folder

