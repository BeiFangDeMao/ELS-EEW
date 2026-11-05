# -*- coding: utf-8 -*-
"""Received samples -> causal preprocessing -> persistent model.stream_step.

Independent stations need independent pipeline instances. The same model weights
may be shared. push() accepts arbitrary packet lengths and emits every 32 points.
There is no full-prefix model.forward call, fake end padding or stored raw history.
"""
from dataclasses import dataclass
import numpy as np
from .causal_preprocessing import CausalWavePreprocessor


@dataclass(frozen=True)
class StreamPrediction:
    end_sample: int
    values: np.ndarray  # (B,4): magnitude, distance/km, sin(azimuth), cos(azimuth)

    @property
    def azimuth_degrees(self):
        return np.degrees(np.arctan2(self.values[:, 2], self.values[:, 3])) % 360.


class CausalStreamingPredictor:
    def __init__(self, model, preprocessor=None, clamp_predictions=True):
        if model.training:
            raise ValueError('Call model.eval() before constructing the pipeline')
        if model.chunk_size != 32 or not callable(getattr(model, 'stream_step', None)):
            raise ValueError('Expected a streaming model with chunk_size=32')
        self.model = model
        self.preprocessor = preprocessor if preprocessor is not None else CausalWavePreprocessor()
        self.clamp_predictions = bool(clamp_predictions)
        self.reset()

    def reset(self):
        """Only for a new independent record/session, never between 32-point blocks."""
        self.preprocessor.reset()
        self.state = None
        self.pending = None
        self.batch_size = None
        self.samples_received = 0
        self.samples_processed = 0
        self.failed = False

    @property
    def pending_samples(self):
        return 0 if self.pending is None else self.pending.shape[-1]

    def push(self, newly_arrived):
        """Return a list of StreamPrediction; [] if fewer than 32 points collected.

        Keep stream identity/order and model device/dtype fixed until reset().
        On a gap/non-finite value/error, this session stops instead of silently
        skipping a block and reusing an invalid continuous-history state.
        """
        import torch
        if self.failed:
            raise RuntimeError('This stream failed; fix the input and reset the session')
        if self.model.training:
            raise RuntimeError('Streaming requires model.eval()')
        x = np.asarray(newly_arrived, dtype=np.float64)
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != 3:
            raise ValueError('Expected newly arrived samples shaped (B,3,N)')
        if self.batch_size is not None and x.shape[0] != self.batch_size:
            raise ValueError('Batch changed; reset or use separate station pipelines')
        if x.shape[-1] == 0:
            return []
        predictions = []
        reference = next(self.model.parameters())
        try:
            if self.batch_size is None:
                self.batch_size = x.shape[0]
                self.pending = np.empty((self.batch_size, 3, 0), dtype=np.float64)
            position = 0
            while position < x.shape[-1]:
                count = min(32 - self.pending_samples, x.shape[-1] - position)
                piece = x[:, :, position:position+count]
                if not np.isfinite(piece).all():
                    raise FloatingPointError('Non-finite received waveform')
                self.pending = np.concatenate((self.pending, piece), axis=-1)
                position += count
                self.samples_received += count
                if self.pending_samples != 32:
                    continue
                filtered = self.preprocessor.step(self.pending)
                wave = torch.from_numpy(filtered).to(device=reference.device, dtype=reference.dtype)
                with torch.inference_mode():
                    pred, new_state = self.model.stream_step(wave, self.state)
                    if tuple(pred.shape) != (self.batch_size, 1, 4) or not torch.isfinite(pred).all().item():
                        raise FloatingPointError('Invalid/non-finite streaming prediction')
                    # Fail on corrupt caches/hidden state before committing them.
                    if not all(torch.isfinite(t).all().item()
                               for t in (*new_state.conv_caches, new_state.gru_hidden)):
                        raise FloatingPointError('Non-finite model state')
                    values = pred[:, 0].cpu().numpy().copy()
                if self.clamp_predictions:
                    values[:, 0] = np.clip(values[:, 0], 0., 10.)
                    values[:, 1] = np.clip(values[:, 1], 0., 200.)
                    values[:, 2:] = np.clip(values[:, 2:], -1., 1.)
                self.state = new_state
                self.samples_processed += 32
                self.pending = np.empty((self.batch_size, 3, 0), dtype=np.float64)
                predictions.append(StreamPrediction(self.samples_processed, values))
        except Exception:
            self.failed = True
            raise
        return predictions
