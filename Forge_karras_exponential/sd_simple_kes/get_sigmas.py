import math
import torch
from torch import linspace, tensor

#source files are from the diffusers library. Modified file to remove error messages on console:
'''
UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  min_inv_rho = tensor(sigma_min, device=device) ** (1 / rho)

'''
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = linspace(0, 1, n, device=device)
    #min_inv_rho = tensor(sigma_min, device=device) ** (1 / rho)
    #max_inv_rho = tensor(sigma_max, device=device) ** (1 / rho)
    def _to_tensor(val, device):
        return val.to(device) if isinstance(val, torch.Tensor) else torch.tensor(val, device=device)

    min_inv_rho = _to_tensor(sigma_min, device) ** (1 / rho)
    max_inv_rho = _to_tensor(sigma_max, device) ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)



def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)