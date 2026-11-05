"""Explicit task adapters; upstream source files remain unchanged."""
import sys
from pathlib import Path
import torch
from torch import nn

PARENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PARENT))
import train_comparison as shared
UPSTREAM=shared.upstream()


class DistPTDistanceOnly(nn.Module):
    """Original TCN and two-value distance head; no invented P-S labels."""
    def __init__(self):
        super().__init__()
        self.backbone=UPSTREAM.distpt_network.DistPT_Network(in_channels=3)
        self.backbone.lin_ptrvl.requires_grad_(False)

    def forward(self,wave):
        distance,_=self.backbone(wave)
        return distance


class BAZNetwork(UPSTREAM.baz_network.BAZ_Network):
    """Same backbone/heads, with per-record rather than cross-batch scaling."""
    def __init__(self):
        super().__init__(in_channels=3,in_samples=288)

    @torch.no_grad()
    def _compute_cov_and_eig(self,wave):
        cov=self._cov(wave)
        # Keep upstream eig ordering and real parts; do not replace with sorted eigh.
        values,vectors=torch.linalg.eig(cov)
        values=values.real.to(wave.dtype).unsqueeze(-1)
        vectors=vectors.real.to(wave.dtype)
        eps=torch.finfo(wave.dtype).eps
        values=values/values.amax(dim=(1,2),keepdim=True).clamp_min(eps)
        cov=cov/cov.abs().amax(dim=(1,2),keepdim=True).clamp_min(eps)
        return torch.cat([cov,values,vectors],dim=-1)


def create_model(name,cfg=None):
    if name=='distpt_distance_only':return DistPTDistanceOnly()
    if name=='baz_network':return BAZNetwork()
    raise ValueError(f'Unknown additional baseline: {name}')


def task_for(name):
    return {'distpt_distance_only':'dis','baz_network':'baz'}[name]


def loss_for(name,output,target):
    if name=='distpt_distance_only':
        return UPSTREAM.MousaviLoss()(output,target)/len(target)
    cosine,sine=output
    angle=torch.deg2rad(target)
    return nn.functional.mse_loss(cosine,angle.cos())+nn.functional.mse_loss(sine,angle.sin())


def prediction_for(name,output):
    if name=='distpt_distance_only':return output[:,:1]
    cosine,sine=output
    return torch.rad2deg(torch.atan2(sine,cosine))%360
