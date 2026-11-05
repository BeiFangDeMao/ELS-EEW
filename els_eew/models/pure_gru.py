"""三分量直接输入两层GRU24，每32点输出；用于卷积前端消融。"""
import torch
from torch import nn

CONFIG = dict(input_dim=3, hidden=24, num_layers=2, dropout=0.1,
              chunk_size=32, mode="pure_gru")


class PureGRUHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(3, 24, 2, batch_first=True, dropout=0.1)
        self.proj = nn.Linear(24, 4)

    def forward(self, wave, state=None):
        output, hidden = self.gru(wave.transpose(1, 2), state)
        return self.proj(output[:, 31::32]), hidden


class PureGRU(nn.Module):
    chunk_size = 32

    def __init__(self):
        super().__init__()
        self.architecture_config = dict(CONFIG)
        # Identity preserves the shared trainer's feature_extractor interface.
        self.feature_extractor = nn.Identity()
        self.output_head = PureGRUHead()
        self.log_sigma_mag = nn.Parameter(torch.tensor(-1.))
        self.log_sigma_dis = nn.Parameter(torch.tensor(4.))
        self.log_sigma_azi = nn.Parameter(torch.tensor(-1.5))

    def _compute(self, wave, state=None):
        if wave.ndim != 3 or wave.size(1) != 3 or wave.size(-1) < 32 or wave.size(-1) % 32:
            raise ValueError("输入需要为(B,3,L)，L为正的32倍数")
        if state is not None:
            if (state.shape != (2, wave.size(0), 24)
                    or state.device != wave.device or state.dtype != wave.dtype):
                raise ValueError("GRU状态与输入不匹配；新记录请重置状态")
        features = self.feature_extractor(wave)
        predictions, hidden = self.output_head(features, state)
        return predictions, features, hidden

    def forward(self, wave):
        predictions, features, _ = self._compute(wave)
        return predictions, features

    @torch.no_grad()
    def stream_step(self, wave_chunk, state=None):
        if self.training:
            raise RuntimeError("流式推理前请调用model.eval()")
        if wave_chunk.ndim != 3 or wave_chunk.shape[1:] != (3, 32):
            raise ValueError("每次流式调用只接受新增的32点")
        reference = self.output_head.gru.weight_ih_l0
        if wave_chunk.device != reference.device or wave_chunk.dtype != reference.dtype:
            raise ValueError("输入的device/dtype必须与模型一致")
        predictions, _, hidden = self._compute(wave_chunk, state)
        return predictions, hidden
