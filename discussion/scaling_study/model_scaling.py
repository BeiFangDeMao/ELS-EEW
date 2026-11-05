"""Reuse the exact CompactGRU components; only width and hidden size vary."""
import math
import time
import numpy as np
import torch
from torch import nn
from models.compact_gru import CompactGRU, WaveformFrontend, GRUOutputHead


def make_model(spec):
    model = CompactGRU()
    w, h = spec['cnn_width'], spec['gru_hidden']
    if (w, h) != (16, 24):
        model.feature_extractor = WaveformFrontend(w, (1, 2, 4, 8))
        model.output_head = GRUOutputHead(w, h)
    model.architecture_config = dict(width=w, hidden=h, kind='ds', dilations=(1, 2, 4, 8), mode='gru')
    return model


def analytical_parameters(w, h):
    return 4*w*w+32*w+3*h*w+9*h*h+16*h+7


def generate_configs():
    selected, candidates = [], []
    for target in (2000, 4000, 8000, 16000, 32000):
        options = []
        for w in range(4, 97, 2):
            for h in range(4, 129, 2):
                n = analytical_parameters(w, h)
                deviation = abs(n-target)/target
                ratio_deviation = abs(math.log((w/h)/(16/24)))
                if deviation <= .08 and ratio_deviation <= .3:
                    options.append(dict(target_parameters=target, cnn_width=w, gru_hidden=h, total_parameters=n,
                                        relative_parameter_error=deviation, width_ratio_error=ratio_deviation,
                                        search_score=deviation+.15*ratio_deviation))
        options.sort(key=lambda x: (x['search_score'], x['total_parameters']))
        if not options:
            raise RuntimeError(f'No reasonable widths for {target}')
        chosen = options[0].copy()
        if target == 8000:
            chosen.update(cnn_width=16, gru_hidden=24, total_parameters=8263,
                          relative_parameter_error=263/8000, width_ratio_error=0., search_score=263/8000)
        name = f'ELS-{target//1000}K'
        chosen.update(model=name, cnn_channels=[chosen['cnn_width']]*4, gru_layers=2, kernel_size=3, dilations=[1, 2, 4, 8])
        selected.append(chosen)
        candidates.extend([dict(model=name, rank=i+1, **item) for i, item in enumerate(options[:8])])
    return selected, candidates


def tensors_in(state):
    if isinstance(state, torch.Tensor):
        yield state
    elif isinstance(state, (tuple, list)):
        for item in state:
            yield from tensors_in(item)


def resources(model, spec):
    count = lambda module: sum(p.numel() for p in module.parameters())
    w, h = spec['cnn_width'], spec['gru_hidden']
    assert model.output_head.gru.num_layers==2 and model.output_head.gru.dropout==.1
    for block,dilation in zip(model.feature_extractor.blocks,(1,2,4,8)):
        conv=block.convs[0].conv
        assert conv.kernel_size==(3,) and conv.dilation==(dilation,) and conv.groups==w
    assert len(model.feature_extractor.blocks)==4
    row = dict(model=spec['model'], cnn_width=w, gru_hidden=h, gru_layers=2,
               total_parameters=count(model), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
               cnn_parameters=count(model.feature_extractor), gru_parameters=count(model.output_head.gru),
               output_parameters=count(model.output_head.proj), task_weight_parameters=3)
    assert row['total_parameters'] == analytical_parameters(w, h)
    assert row['total_parameters'] == sum(row[k] for k in ('cnn_parameters', 'gru_parameters', 'output_parameters', 'task_weight_parameters'))
    model = model.cpu().eval()
    counts = dict(cnn=0, gru=0, output=0)
    handles = []
    # Hook the REAL incremental stream_step, not the full-record forward.
    def conv_hook(module, inputs, output):
        counts['cnn'] += output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]
    def gru_hook(module, inputs, output):
        batch, steps = inputs[0].shape[:2]
        assert steps == 32, 'GRU must update on all 32 new samples'
        for layer in range(module.num_layers):
            ins = module.input_size if layer == 0 else module.hidden_size
            counts['gru'] += batch*steps*3*module.hidden_size*(ins+module.hidden_size)
    def linear_hook(module, inputs, output):
        counts['output'] += output.numel()*module.in_features
    for module in model.modules():
        if isinstance(module, nn.Conv1d): handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.GRU): handles.append(module.register_forward_hook(gru_hook))
        elif isinstance(module, nn.Linear): handles.append(module.register_forward_hook(linear_hook))
    try:
        with torch.inference_mode():
            _, state = model.stream_step(torch.zeros(1, 3, 32))
    finally:
        for handle in handles: handle.remove()
    conv_numel = sum(t.numel() for t in tensors_in(state[0]))
    gru_numel = state[1][0].numel()
    assert conv_numel == 30*w and gru_numel == 2*h
    assert counts == dict(cnn=32*(4*w*w+15*w), gru=32*(3*h*w+9*h*h), output=4*h)
    row.update(cnn_macs=counts['cnn'], gru_macs=counts['gru'], output_macs=counts['output'],
               macs_per_update=sum(counts.values()), fp32_weights_kib=4*row['total_parameters']/1024,
               inference_weights_kib=4*(row['total_parameters']-3)/1024,
               conv_cache_kib=4*conv_numel/1024, gru_hidden_kib=4*gru_numel/1024,
               persistent_state_kib=4*(conv_numel+gru_numel)/1024)
    return row


def validate_streaming(model, wave):
    """Compare identical causal computations at controlled FP32 precision.

    cuDNN TF32 convolution algorithms can differ between a 2976-point call and
    93 calls of 32 points. Scope the precision change to this diagnostic only;
    training, normal evaluation, weights and original tolerances are unchanged.
    """
    model.eval()
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_rng = torch.get_rng_state()
    old_cuda_rng = torch.cuda.get_rng_state_all() if wave.is_cuda else None
    cuda = wave.is_cuda
    try:
        if cuda: torch.backends.cuda.matmul.allow_tf32 = False
        # Preserve enabled/benchmark/deterministic flags: cudnn.flags defaults
        # enabled=False, so using only allow_tf32=False would disable cuDNN!
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                allow_tf32=False if cuda else torch.backends.cudnn.allow_tf32):
            with torch.inference_mode():
                full, _ = model(wave)
                state, pieces = None, []
                for chunk in wave.split(32, dim=-1):
                    output, state = model.stream_step(chunk, state)
                    pieces.append(output)
                streamed = torch.cat(pieces, dim=1)
                if not torch.isfinite(full).all() or not torch.isfinite(streamed).all():
                    raise FloatingPointError('Nonfinite full-window/streaming predictions')
                torch.testing.assert_close(full, streamed, rtol=2e-4, atol=2e-5)
                changed = wave.clone()
                changed[:, :, 96:] = 100*torch.randn_like(changed[:, :, 96:])
                altered, _ = model(changed)
                torch.testing.assert_close(full[:, :3], altered[:, :3], rtol=1e-5, atol=1e-6)
    finally:
        if cuda: torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.set_rng_state(old_rng)
        if old_cuda_rng is not None: torch.cuda.set_rng_state_all(old_cuda_rng)
    assert full.shape == (len(wave), wave.shape[-1]//32, 4) and torch.isfinite(full).all()
    return dict(max_stream_difference=float((full-streamed).abs().max()),
                max_difference_by_output=(full-streamed).abs().amax(dim=(0,1)).cpu().tolist(),
                prefix_invariance=True, diagnostic_tf32_disabled=cuda,
                precision_flags_and_rng_restored=True, rtol=2e-4, atol=2e-5)


def benchmark_cpu(model, steps=1000, warmup=100, threads=1):
    """Optional actual batch=1 timing; never inferred from MACs, no preprocessing."""
    old = torch.get_num_threads()
    model = model.cpu().eval()
    x, state = torch.zeros(1, 3, 32), None
    values = []
    try:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            for i in range(warmup+steps):
                start = time.perf_counter_ns()
                _, state = model.stream_step(x, state)
                elapsed = (time.perf_counter_ns()-start)/1e6
                if i >= warmup: values.append(elapsed)
    finally:
        torch.set_num_threads(old)
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p90_ms=float(np.percentile(values,90)),
                threads=threads, steps=steps, batch_size=1, preprocessing_included=False)
