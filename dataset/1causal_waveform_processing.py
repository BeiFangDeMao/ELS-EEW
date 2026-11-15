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


# Input is already P-aligned, calibrated in gal and resampled to 100 Hz.
# No arrival annotations, raw-file reader or P-wave detection is required.
from pathlib import Path
import argparse
import json

INPUT_H5 = Path(__file__).resolve().parent / 'p_aligned_30s_100hz.h5'
OUTPUT_H5 = Path(__file__).resolve().parent / 'p_aligned_causal_100hz.h5'
TARGET_LENGTH = 2976  # Exact training length; use 3000 for the complete 30 s.


def process_waveform(wave, target_length=TARGET_LENGTH, chunk_size=32):
    """(N,3), Z/N/E, gal, 100Hz -> causal float32 (target_length,3).

    Each call initializes an independent record. For continuous streaming,
    keep a CausalWavePreprocessor instance and call its step without resetting.
    """
    x = np.asarray(wave)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < target_length:
        raise ValueError('Require (N,3), N >= target_length; no zero padding')
    if target_length < 1 or chunk_size < 1 or not np.isfinite(x).all():
        raise ValueError('Invalid length or non-finite input')
    processor = CausalWavePreprocessor()
    parts = [processor.step(x[i:min(i+chunk_size,target_length)].T[None])[0].T
             for i in range(0,target_length,chunk_size)]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def process_h5(source, destination, length=TARGET_LENGTH):
    """Direct H5 datasets (N,3); copies only mag/dis/azi, never arrival picks."""
    import h5py
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    with h5py.File(source,'r') as src:
        if bool(src.attrs.get('causal_processed',False)):
            raise ValueError('Input already processed')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(destination,'x') as dst:
            dst.attrs.update(causal_processed=True,sampling_freq_hz=100.,
                             channel_order='Z,N,E',unit='gal',
                             processing=json.dumps(CausalWavePreprocessor().describe()))
            for i,name in enumerate(sorted(src.keys()),1):
                ds=src[name]
                if not isinstance(ds,h5py.Dataset): raise ValueError('Require direct waveform datasets')
                if bool(ds.attrs.get('causal_processed',False)): raise ValueError('Already processed')
                rate=float(ds.attrs.get('sampling_freq_hz',src.attrs.get('sampling_freq_hz',np.nan)))
                if not np.isclose(rate,100.,rtol=0,atol=1e-6): raise ValueError('Input must already be 100Hz')
                for attr,root_attr,expected in [('channel_order','channel_order','Z,N,E'),('waveform_unit','unit','gal')]:
                    value=ds.attrs.get(attr,src.attrs.get(root_attr))
                    if isinstance(value,bytes): value=value.decode('utf-8')
                    if value!=expected: raise ValueError(f'{name}: require {attr}={expected}')
                output=dst.create_dataset(name,data=process_waveform(ds[:],length))
                for label in ('mag','dis','azi'):
                    if label in ds.attrs: output.attrs[label]=ds.attrs[label]
                output.attrs.update(sampling_freq_hz=100.,channel_order='Z,N,E',waveform_unit='gal',
                                    sample_count=length,window_duration_s=length/100.,causal_processed=True)
                if i%1000==0: print(f'Processed {i} records',flush=True)
    print(f'Saved: {destination}')


def self_test():
    x=np.random.default_rng(42).normal(size=(3000,3))
    a=process_waveform(x)
    full=CausalWavePreprocessor().step(x[:2976].T[None])[0].T
    np.testing.assert_array_equal(a,full)
    changed=x.copy();changed[1000:]+=100
    np.testing.assert_array_equal(a[:1000],process_waveform(changed)[:1000])
    assert process_waveform(x,3000).shape==(3000,3)
    print('PASS: full/block equivalence, prefix causality and 2976/3000 lengths')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Causal processing of already P-aligned Z/N/E waveforms')
    parser.add_argument('--input',type=Path,default=INPUT_H5)
    parser.add_argument('--output',type=Path,default=OUTPUT_H5)
    parser.add_argument('--length',type=int,default=TARGET_LENGTH)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test()
    else: process_h5(args.input,args.output,args.length)
