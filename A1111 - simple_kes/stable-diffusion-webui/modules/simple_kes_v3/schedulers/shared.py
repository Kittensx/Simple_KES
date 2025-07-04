import math
import torch
from torch import linspace, tensor


valid_decay_patterns = [
    'zero', 'geometric', 'harmonic', 'logarithmic',
    'extrapolate', 'fractional', 'exponential', 'linear'
]
valid_decay_modes = [
    'append',
    'blend',
    'replace'
]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def apply_last_tail(sigmas, device, decay_pattern='zero'):
    """
    Applies a single final sigma step using context-aware dynamic decay based on the sequence trend.
    """
    if decay_pattern == 'zero':
        return append_zero(sigmas)

    last_sigma = sigmas[-1]

    if len(sigmas) >= 2:
        deltas = torch.abs(sigmas[:-1] - sigmas[1:])
        avg_delta = torch.mean(deltas).item()
    else:
        avg_delta = last_sigma.item() * 0.1

    if decay_pattern == 'geometric':
        dynamic_decay_rate = max(1 - (avg_delta / (last_sigma + 1e-5)), 0.85)
        next_sigma = max(last_sigma * dynamic_decay_rate, 1e-5)

    elif decay_pattern == 'harmonic':
        next_sigma = max(last_sigma - avg_delta, 1e-5)

    elif decay_pattern == 'logarithmic':
        next_sigma = max(last_sigma - (avg_delta / math.log(len(sigmas) + 2)), 1e-5)

    elif decay_pattern == 'extrapolate':
        if len(sigmas) >= 2:
            last_delta = sigmas[-2] - sigmas[-1]
        else:
            last_delta = avg_delta
        next_sigma = max(last_sigma - last_delta, 1e-5)

    elif decay_pattern == 'fractional':
        next_sigma = max(last_sigma * 0.1, 1e-5)

    elif decay_pattern == 'exponential':
        next_sigma = max(last_sigma * math.exp(-avg_delta), 1e-5)

    elif decay_pattern == 'linear':
        next_sigma = max(last_sigma - (avg_delta * 0.5), 1e-5)

    else:
        raise ValueError(f"Unknown decay pattern: {decay_pattern}. Valid decay patterns are: {valid_decay_patterns}")

    tail_tensor = torch.tensor([next_sigma], device=device)
    return torch.cat([sigmas, tail_tensor])


def apply_decay_tail(sigmas, device, decay_pattern='geometric', tail_steps=5):
    """
    Applies a context-aware multi-step decay tail based on the progression of the entire sigma sequence.
    """
    tail = []

    if decay_pattern == 'zero':
        return append_zero(sigmas)

    if len(sigmas) >= 2:
        deltas = torch.abs(sigmas[:-1] - sigmas[1:])
        avg_delta = torch.mean(deltas).item()
    else:
        avg_delta = sigmas[-1].item() * 0.1

    last_sigma = sigmas[-1]

    for step in range(tail_steps):
        if decay_pattern == 'geometric':
            dynamic_decay_rate = max(1 - (avg_delta / (last_sigma + 1e-5)), 0.85)
            next_sigma = max(last_sigma * dynamic_decay_rate, 1e-5)

        elif decay_pattern == 'harmonic':
            next_sigma = max(last_sigma - (avg_delta / (step + 1)), 1e-5)

        elif decay_pattern == 'logarithmic':
            next_sigma = max(last_sigma - (avg_delta / math.log(len(sigmas) + step + 2)), 1e-5)

        elif decay_pattern == 'extrapolate':
            if len(sigmas) >= 2:
                last_delta = sigmas[-2] - sigmas[-1]
            else:
                last_delta = avg_delta
            next_sigma = max(last_sigma - last_delta, 1e-5)

        elif decay_pattern == 'fractional':
            next_sigma = max(last_sigma * 0.1, 1e-5)

        elif decay_pattern == 'exponential':
            next_sigma = max(last_sigma * math.exp(-avg_delta * (step + 1)), 1e-5)

        elif decay_pattern == 'linear':
            next_sigma = max(last_sigma - (avg_delta * 0.5 * (step + 1)), 1e-5)

        else:
            raise ValueError(f"Unknown decay pattern: {decay_pattern}. Valid decay patterns are: {valid_decay_patterns}")

        tail.append(next_sigma)
        last_sigma = next_sigma

    tail_tensor = torch.tensor(tail, device=device)
    return torch.cat([sigmas, tail_tensor])


      

def blend_decay_tail(sigmas, device, decay_pattern='geometric', tail_steps=5):
    """
    Applies in-place blending on the last N steps using decay patterns.
    """
    for i in range(1, tail_steps + 1):
        idx = -i
        base_sigma = sigmas[idx]

        if len(sigmas) >= 2:
            deltas = torch.abs(sigmas[:-1] - sigmas[1:])
            avg_delta = torch.mean(deltas).item()
        else:
            avg_delta = base_sigma.item() * 0.1
        if decay_pattern == 'zero':
            sigmas[idx] = 0.0
            continue
        if decay_pattern == 'geometric':
            dynamic_decay_rate = max(1 - (avg_delta / (base_sigma + 1e-5)), 0.85)
            new_sigma = max(base_sigma * dynamic_decay_rate, 1e-5)

        elif decay_pattern == 'harmonic':
            new_sigma = max(base_sigma - (avg_delta / i), 1e-5)

        elif decay_pattern == 'logarithmic':
            new_sigma = max(base_sigma - (avg_delta / math.log(len(sigmas) + 2 - i)), 1e-5)

        elif decay_pattern == 'extrapolate':
            if len(sigmas) >= 2:
                last_delta = sigmas[-2] - sigmas[-1]
            else:
                last_delta = avg_delta
            new_sigma = max(base_sigma - last_delta, 1e-5)

        elif decay_pattern == 'fractional':
            new_sigma = max(base_sigma * 0.1, 1e-5)

        elif decay_pattern == 'exponential':
            new_sigma = max(base_sigma * math.exp(-avg_delta), 1e-5)

        elif decay_pattern == 'linear':
            new_sigma = max(base_sigma - (avg_delta * 0.5), 1e-5)

        else:
            raise ValueError(f"Unknown decay pattern: {decay_pattern}. Valid decay patterns are: {valid_decay_patterns}")

        sigmas[idx] = (base_sigma + new_sigma) / 2

    return sigmas


def replace_tail(sigmas, device, decay_pattern='geometric', tail_steps=5):
    available_steps = len(sigmas)

    # Clamp tail_steps to the number of available steps
    if tail_steps >= available_steps:
        print(f"[Replace Tail] Requested {tail_steps} steps, but only {available_steps} available. Adjusting to {available_steps - 1} steps.")
        tail_steps = available_steps - 1  # Ensure we leave at least one sigma

    sigmas = sigmas[:-tail_steps]  # Remove the last N steps
    return apply_decay_tail(sigmas, device, decay_pattern, tail_steps)

