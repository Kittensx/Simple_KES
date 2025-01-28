import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential
import numpy as np
import yaml
import os
import random
import warnings
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    '''
    Parameters:
        n (int): Number of steps.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.
        device (torch.device): The device on which to perform computations (e.g., 'cuda' or 'cpu').
        start_blend (float): Initial blend factor for dynamic blending.
        end_bend (float): Final blend factor for dynamic blending.
        sharpen_factor (float): Sharpening factor to be applied adaptively.
        early_stopping_threshold (float): Threshold to trigger early stopping.
        update_interval (int): Interval to update blend factors.
        initial_step_size (float): Initial step size for adaptive step size calculation.
        final_step_size (float): Final step size for adaptive step size calculation.
        initial_noise_scale (float): Initial noise scale factor.
        final_noise_scale (float): Final noise scale factor.
        step_size_factor: Adjust to compensate for oversmoothing
        noise_scale_factor: Adjust to provide more variation
    '''
    
    n: int
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    user_config: dict = None   
    
    debug: bool = False  
    sigma_min: float = 0.01
    sigma_max: float = 50.0
    start_blend: float = 0.1
    end_blend: float = 0.5
    sharpness: float = 0.95
    early_stopping_threshold: float = 0.01
    update_interval: int = 10
    initial_step_size: float = 0.9
    final_step_size: float = 0.2
    initial_noise_scale: float = 1.25
    final_noise_scale: float = 0.8
    smooth_blend_factor: int = 11
    step_size_factor: float = 0.8
    noise_scale_factor: float = 0.9
    randomize: bool = False
    
    # Randomization parameters with defaults
    sigma_min_rand: bool = False
    sigma_min_rand_min: float = 0.001
    sigma_min_rand_max: float = 0.05

    sigma_max_rand: bool = False
    sigma_max_rand_min: float = 0.05
    sigma_max_rand_max: float = 0.20

    start_blend_rand: bool = False
    start_blend_rand_min: float = 0.05
    start_blend_rand_max: float = 0.2

    end_blend_rand: bool = False
    end_blend_rand_min: float = 0.4
    end_blend_rand_max: float = 0.6

    sharpness_rand: bool = False
    sharpness_rand_min: float = 0.85
    sharpness_rand_max: float = 1.0

    early_stopping_rand: bool = False
    early_stopping_rand_min: float = 0.001
    early_stopping_rand_max: float = 0.02

    update_interval_rand: bool = False
    update_interval_rand_min: int = 5
    update_interval_rand_max: int = 10

    initial_step_rand: bool = False
    initial_step_rand_min: float = 0.7
    initial_step_rand_max: float = 1.0

    final_step_rand: bool = False
    final_step_rand_min: float = 0.1
    final_step_rand_max: float = 0.3

    initial_noise_rand: bool = False
    initial_noise_rand_min: float = 1.0
    initial_noise_rand_max: float = 1.5

    final_noise_rand: bool = False
    final_noise_rand_min: float = 0.6
    final_noise_rand_max: float = 1.0

    smooth_blend_factor_rand: bool = False
    smooth_blend_factor_rand_min: int = 6
    smooth_blend_factor_rand_max: int = 11

    step_size_factor_rand: bool = False
    step_size_factor_rand_min: float = 0.65
    step_size_factor_rand_max: float = 0.85

    noise_scale_factor_rand: bool = False
    noise_scale_factor_rand_min: float = 0.75
    noise_scale_factor_rand_max: float = 0.95
    
   

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Create an instance of SchedulerConfig from a dictionary.
        Unknown fields are ignored.
        """
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**filtered_config)
        
    '''
    How to Use:
    default_config = {
        "debug": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sigma_min": 0.01,
        "sigma_max": 50.0,
        "randomize": True,
        "sigma_min_rand": True,
        "sigma_min_rand_min": 0.002,
        "sigma_min_rand_max": 0.04
    }

    # Create a config instance using the dictionary
    scheduler_config = SchedulerConfig.from_dict(default_config)

    print(scheduler_config)
    '''

class SimpleKarrasExponentialScheduler(SchedulerMixin):
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, config_path="modules\kes_config\simple_kes_scheduler.yaml", user_config=None, n=None):
        super().__init__()        
        self.scheduler_config = SchedulerConfig(n=n)
        self.config_path = config_path       
        config_dict = self.load_config('scheduler', {})
        self.config = SchedulerConfig.from_dict(config_dict)
        if self.config.user_config:
            for key, value in self.config.user_config.items():
                if key in self.config.__dataclass_fields__:
                    setattr(self.config, key, value)

        # Set scheduler parameters from config or defaults
        self.num_train_timesteps = getattr(self.config, 'n', num_train_timesteps)
        self.beta_start = getattr(self.config, 'beta_start', beta_start)
        self.beta_end = getattr(self.config, 'beta_end', beta_end)
        self.timesteps = torch.linspace(0, 1, self.num_train_timesteps)
        self.sigmas = self.get_sigmas()       
	    
    def load_config(self, section, default):
        try:
            with open(self.config_path, "r") as f:
                full_config = yaml.safe_load(f) or {}
                return full_config.get(section, default)
        except FileNotFoundError:
            print(f"Warning: Configuration file {self.config_path} not found. Using default values.")
            return default

    def get_sigmas(self, n=None):
        """Generate sigmas based on exponential Karras schedule."""
        # Expand sigma_max slightly to account for smoother transitions
        n = n if n is not None else getattr(self.config, 'n', self.num_train_timesteps)    
        sigma_max = self.config.sigma_max * 1.1
        sigma_min = self.config.sigma_min        
        
        betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_train_timesteps) ** 2
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = (1.0 - alpha_cumprod) / (alpha_cumprod**0.5)
        if self.config.sigma_min >= self.config.sigma_max:
            raise ValueError("sigma_min should be less than sigma_max. Check configuration.")
           

        # Handle potential NaN or infinite values
        if torch.isnan(sigmas).any() or torch.isinf(sigmas).any():
            raise ValueError("Invalid values detected in sigmas. Check scheduler settings.")
        
        print(f"Generating sigmas with steps: {n}, sigma_min: {self.config.sigma_min}, sigma_max: {self.config.sigma_max}")
        print(f"Generated sigmas: {sigmas}")

        return sigmas

    def set_timesteps(self, num_inference_steps, n=None):
        """Set timesteps for inference based on the configured number of steps."""
        n = n if n is not None else getattr(self.config, 'n', self.num_inference_steps)  
        self.timesteps = torch.linspace(0, self.num_train_timesteps - 1, num_inference_steps).to(torch.int64)
        self.sigmas = self.get_sigmas()[self.timesteps]

    def step(self, model_output, timestep, sample, n=None, **kwargs):
        """
        Perform one step in the denoising process.

        Args:
            model_output: The output from the model (latent tensor).
            timestep: The current timestep.
            sample: The current noisy sample.
        
        Returns:
            Denoised sample after applying the custom scheduler step.
        """
        n = n if n is not None else getattr(self.config, 'n', self.timestep)  
        sigma = self.sigmas[timestep]

        # Apply noise correction based on the current timestep sigma
        denoised_sample = sample - sigma * model_output

        return denoised_sample

    def scale_model_input(self, sample, timestep, n=None):
        """Scale the input sample according to the sigma schedule."""
        n = n if n is not None else getattr(self.config, 'n', self.timestep) 
        return sample / (1.0 + self.sigmas[timestep])
    
    @staticmethod
    def get_random_or_default(config, key_prefix, default_value, global_randomize):
        """Helper function to either randomize a value based on conditions or return the default."""

        # Dynamically construct the attribute names from the dataclass fields
        randomize_flag = global_randomize or getattr(config, f"{key_prefix}_rand", False)

        # Set more meaningful ranges based on parameter type
        if key_prefix == "sigma_min":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.01)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.1)
        elif key_prefix == "sigma_max":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 10)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 60)
        elif key_prefix == "start_blend":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.05)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.20)
        elif key_prefix == "end_blend":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.4)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.6)
        elif key_prefix == "sharpness":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.8)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 1.0)
        elif key_prefix == "early_stopping_threshold":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.01)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.05)
        elif key_prefix == "update_interval":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 5)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 15)
        elif key_prefix == "initial_step_size":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.5)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 1.0)
        elif key_prefix == "final_step_size":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.1)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.3)
        elif key_prefix == "initial_noise_scale":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 1.0)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 1.5)
        elif key_prefix == "final_noise_scale":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.6)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 1.0)
        elif key_prefix == "smooth_blend_factor":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 6)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 11)
        elif key_prefix == "step_size_factor":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.65)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.85)
        elif key_prefix == "noise_scale_factor":
            rand_min = getattr(config, f"{key_prefix}_rand_min", 0.75)
            rand_max = getattr(config, f"{key_prefix}_rand_max", 0.95)            
        else:
            # Generic fallback when no specific range is provided
            rand_min = getattr(config, f"{key_prefix}_rand_min", default_value * 0.75)
            rand_max = getattr(config, f"{key_prefix}_rand_max", default_value * 1.25)
        if randomize_flag:
            return random.uniform(rand_min, rand_max)
        return default_value   
    
    def simple_karras_exponential_scheduler(self, n=None, steps=None, device=None, user_config=None, **kwargs):    
        """
        Scheduler function that blends sigma sequences using Karras and Exponential methods with adaptive parameters.

        Parameters:
            n (int): Number of steps.
            sigma_min (float): Minimum sigma value.
            sigma_max (float): Maximum sigma value.
            device (torch.device): The device on which to perform computations (e.g., 'cuda' or 'cpu').
            start_blend (float): Initial blend factor for dynamic blending.
            end_bend (float): Final blend factor for dynamic blending.
            sharpen_factor (float): Sharpening factor to be applied adaptively.
            early_stopping_threshold (float): Threshold to trigger early stopping.
            update_interval (int): Interval to update blend factors.
            initial_step_size (float): Initial step size for adaptive step size calculation.
            final_step_size (float): Final step size for adaptive step size calculation.
            initial_noise_scale (float): Initial noise scale factor.
            final_noise_scale (float): Final noise scale factor.
            step_size_factor: Adjust to compensate for oversmoothing
            noise_scale_factor: Adjust to provide more variation
            
        Returns:
            torch.Tensor: A tensor of blended sigma values.
        """
        # Use provided steps if given, otherwise default to config or fallback value
        n = steps if steps is not None else getattr(self.config, 'n', 30)        
       
        device = "cuda" if device is not None else getattr(self.config, 'device', "cuda")
       
        
        # Generate sigma sequences using Karras and Exponential methods       
        sigmas_karras = get_sigmas_karras(n=n, sigma_min=self.config.sigma_min, sigma_max=self.config.sigma_max, device=device)
        sigmas_exponential = get_sigmas_exponential(n=n, sigma_min=self.config.sigma_min, sigma_max=self.config.sigma_max, device=self.config.device)

             
        # Match lengths of sigma sequences
        target_length = min(len(sigmas_karras), len(sigmas_exponential))  
        sigmas_karras = sigmas_karras[:target_length]
        sigmas_exponential = sigmas_exponential[:target_length]
        # Ensure the final length matches `n`
        if len(sigmas_karras) < n:
            # If sequences are shorter than `n`, pad with the last value
            padding_karras = torch.full((n - len(sigmas_karras),), sigmas_karras[-1]).to(sigmas_karras.device)
            sigmas_karras = torch.cat([sigmas_karras, padding_karras])

            padding_exponential = torch.full((n - len(sigmas_exponential),), sigmas_exponential[-1]).to(sigmas_exponential.device)
            sigmas_exponential = torch.cat([sigmas_exponential, padding_exponential])
        elif len(sigmas_karras) > n:
            # If sequences are longer than `n`, truncate to `n`
            sigmas_karras = sigmas_karras[:n]
            sigmas_exponential = sigmas_exponential[:n]
        
        if sigmas_karras is None:
            raise ValueError("Sigmas Karras:{sigmas_karras} Failed to generate or assign sigmas correctly.")
        if sigmas_exponential is None:    
            raise ValueError("Sigmas Exponential: {sigmas_exponential} Failed to generate or assign sigmas correctly.")              
        try:
            pass
        except Exception as e:
            print(f"An Exception {e} occurred.")
        
        # Define progress and initialize blend factor
        progress = torch.linspace(0, 1, len(sigmas_karras)).to(device)
       
        sigs = torch.zeros_like(sigmas_karras).to(device)
        
        # Iterate through each step, dynamically adjust blend factor, step size, and noise scaling
        for i in range(len(sigmas_karras)):
            # Adaptive step size and blend factor calculations
            step_size = self.config.initial_step_size * (1 - progress[i]) + self.config.final_step_size * progress[i] * self.config.step_size_factor
            dynamic_blend_factor = self.config.start_blend * (1 - progress[i]) + self.config.end_blend * progress[i]
            noise_scale = self.config.initial_noise_scale * (1 - progress[i]) + self.config.final_noise_scale * progress[i] * self.config.noise_scale_factor
       
            # Calculate smooth blending between the two sigma sequences
            smooth_blend = torch.sigmoid((dynamic_blend_factor - 0.5) * self.config.smooth_blend_factor)
       
            # Compute blended sigma values
            blended_sigma = sigmas_karras[i] * (1 - smooth_blend) + sigmas_exponential[i] * smooth_blend
           
            # Apply step size and noise scaling
            sigs[i] = blended_sigma * step_size * noise_scale

        # Optional: Adaptive sharpening based on sigma values
        sharpen_mask = torch.where(sigs < self.config.sigma_min * 1.5, self.config.sharpness, 1.0).to(device)
        
        sigs = sigs * sharpen_mask
        
        # Implement early stop criteria based on sigma convergence
        change = torch.abs(sigs[1:] - sigs[:-1])
        if torch.all(change < self.config.early_stopping_threshold):
            #custom_logger.info("Early stopping criteria met."   )
            return sigs[:len(change) + 1].to(device)
        
        if torch.isnan(sigs).any() or torch.isinf(sigs).any():
            raise ValueError("Invalid sigma values detected (NaN or Inf).")

        return sigs.to(device)
