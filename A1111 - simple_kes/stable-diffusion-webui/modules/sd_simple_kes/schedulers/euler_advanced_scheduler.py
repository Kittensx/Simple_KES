import torch
from modules.sd_simple_kes.schedulers.shared import apply_last_tail, apply_decay_tail, valid_decay_patterns, valid_decay_modes, blend_decay_tail, replace_tail
import math

def get_sigmas_euler(steps, sigma_min, sigma_max, device='cpu'):
    """
    Sigma schedule designed to work well with Euler sampling.
    Logarithmic spacing with a smooth transition.
    """
    def _to_tensor(val, device):
        return val.to(device) if isinstance(val, torch.Tensor) else torch.tensor(val, device=device)

    # Convert sigma_min and sigma_max to tensors safely
    sigma_min = _to_tensor(sigma_min, device)
    sigma_max = _to_tensor(sigma_max, device)

    sigmas = torch.exp(torch.linspace(math.log(sigma_max.item()), math.log(sigma_min.item()), steps, device=device))
    
    tails = None
    decay = None

    
    return tails, decay, sigmas
    
def get_sigmas_euler_advanced(steps, sigma_min, sigma_max, device='cpu', blend_factor=0.5, decay_pattern=None, decay_mode=None, tail_steps=None):
    def _to_tensor(val, device):
        return val.to(device) if isinstance(val, torch.Tensor) else torch.tensor(val, device=device)
    ramp = torch.linspace(0, 1, steps, device=device)
    sigmas_exp = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, device=device))
    sigmas_karras = (sigma_max ** (1 / 7) + ramp * (sigma_min ** (1 / 7) - sigma_max ** (1 / 7))) ** 7

    # Blend Karras and Exponential schedules for a hybrid Euler-friendly progression
    sigmas = (1 - blend_factor) * sigmas_exp + blend_factor * sigmas_karras
    tails = None
    decay = None

    if decay_pattern:
        if decay_pattern in valid_decay_patterns:
            tails = apply_last_tail(sigmas, device, decay_pattern)
        elif decay_pattern not in valid_decay_patterns:
            print(f"[Warning] decay_pattern: {decay_pattern} not in valid decay patterns: {valid_decay_patterns}") 
    if decay_mode:
        if decay_mode in valid_decay_modes:
            if 'append':
                apply_decay_tail(sigmas, device, decay_pattern)
            if 'blend':
                blend_decay_tail(sigmas, device, decay_pattern, tail_steps)
            if 'replace':
                replace_tail(sigmas, device, decay_pattern, tail_steps)                   
        elif decay_mode not in valid_decay_modes:
            print(f"[Warning] decay_mode: {decay_mode} not in valid decay modes: {valid_decay_modes}")
    return tails, decay, sigmas