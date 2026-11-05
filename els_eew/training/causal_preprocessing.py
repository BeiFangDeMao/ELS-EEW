# -*- coding: utf-8 -*-
"""Incremental Z/N/E preprocessing. Input and output are (B,3,N), in gal.

No future statistics, linear detrend, backward filtering or normalization.
Each instance belongs to fixed ordered streams; reset when those streams change.
Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
"""
import math
import numpy as np
from scipy.signal import butter, sosfilt


class CausalWavePreprocessor:
    def __init__(self, sampling_rate=100., band=(.75, 40.), order=4,
                 mode='causal_bandpass', subtract_initial=True):
        if mode not in ('causal_bandpass', 'raw'):
            raise ValueError("mode must be 'causal_bandpass' or 'raw'")
        if not math.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError('sampling_rate must be positive and finite')
        if (len(band) != 2 or not all(math.isfinite(x) for x in band)
                or not 0 < band[0] < band[1] < sampling_rate / 2):
            raise ValueError('Expected 0 < low < high < sampling_rate/2')
        if not isinstance(order, int) or isinstance(order, bool) or order < 1:
            raise ValueError('order must be a positive integer')
        self.sampling_rate = float(sampling_rate)
        self.band = tuple(map(float, band))
        self.order = order
        self.mode = mode
        self.subtract_initial = bool(subtract_initial) if mode != 'raw' else False
        # N=4 bandpass has total order 8 (4 second-order sections), matching
        # the old butter(4, [low,high], btype='bandpass') design convention.
        self.sos = (butter(order, self.band, btype='bandpass', fs=sampling_rate,
                           output='sos') if mode != 'raw' else None)
        self.reset()

    def reset(self):
        self.offset = None
        self.zi = None
        self.batch_size = None
        self.samples_seen = 0

    def step(self, newly_arrived):
        """Process only this block; carry final SOS delays into the next block.

        Fixed first-sample subtraction uses no block mean. High-pass filtering
        suppresses slow drift but does not reproduce full-record linear detrend.
        Float64 filter state is retained; only the model input becomes float32.
        """
        x = np.asarray(newly_arrived, dtype=np.float64)
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] != 3 or x.shape[2] < 1:
            raise ValueError('Expected newly arrived samples shaped (B,3,N), N>0')
        if self.batch_size is not None and x.shape[0] != self.batch_size:
            raise ValueError('Stream batch changed; reset all states first')
        if not np.isfinite(x).all():
            raise FloatingPointError('Non-finite newly arrived waveform; reset after resolving the gap')
        initial = self.batch_size is None
        offset = x[:, :, :1].copy() if initial else self.offset
        if self.mode == 'raw':
            y, next_zi = x, None
        else:
            zi = (np.zeros((len(self.sos), x.shape[0], 3, 2), dtype=np.float64)
                  if initial else self.zi)
            y, next_zi = sosfilt(self.sos, x - offset if self.subtract_initial else x,
                                axis=-1, zi=zi)
        with np.errstate(over='ignore', invalid='ignore'):
            output = np.ascontiguousarray(y, dtype=np.float32)
        if not np.isfinite(output).all() or (next_zi is not None and not np.isfinite(next_zi).all()):
            raise FloatingPointError('Non-finite filtered data or filter state; do not continue this stream')
        # Commit only after validating this step; failed input leaves old state intact.
        self.batch_size, self.offset, self.zi = x.shape[0], offset, next_zi
        self.samples_seen += x.shape[2]
        return output

    def describe(self):
        return dict(mode=self.mode, sampling_rate_hz=self.sampling_rate,
                    band_hz=self.band if self.sos is not None else None,
                    butter_design_order=self.order if self.sos is not None else None,
                    total_bandpass_order=2*self.order if self.sos is not None else None,
                    subtract_first_arrived_sample=self.subtract_initial,
                    filter_state_dtype='float64', model_input_dtype='float32',
                    normalization=False, linear_detrend=False, backward_filtering=False,
                    delay_compensation=False, amplitude_unit='gal')
