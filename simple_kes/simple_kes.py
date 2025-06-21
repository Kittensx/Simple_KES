import torch
import logging
from get_sigmas import get_sigmas_karras, get_sigmas_exponential 
import os
import yaml
import random

from datetime import datetime
import warnings
import os
import logging
from datetime import datetime

import math
from typing import Optional

import torch

# User config file is necessary for this to work!

    
class SimpleKEScheduler:
    """
    SimpleKEScheduler
    ------------------
    A hybrid scheduler that combines Karras-style sigma sampling
    with exponential decay and blending controls. Supports parameterized
    customization for use in advanced diffusion pipelines.

    Parameters:
    - steps (int): Number of inference steps.
    - device (torch.device): Target device (e.g. 'cuda').
    - config (dict): Scheduler-specific configuration options.

    Usage:
        scheduler = SimpleKEScheduler(steps=30, device='cuda', config=config_dict)
        sigmas = scheduler.get_sigmas()
    """

    def __init__(self, steps: int, device: torch.device, config: dict):             
        self.steps = steps 
        self.device = device
       

        # Load config
        if isinstance(config, str):
            self.config_path = os.path.abspath(os.path.normpath(config))
            self.config_data = self.load_config()
            
        elif isinstance(config, dict):
            self.config_path = None
            self.config_data = config
        else:
            raise TypeError(f"[KEScheduler] Invalid config type: {type(config)}")

        # Use a fresh working config
        self.config = self.config_data.copy()
        self.settings = self.config.copy() 

        
        # ✅ Map aliases → internal names        
        for key, value in self.settings.items():
            
            setattr(self, key, value)
            
        if self.settings.get("global_randomize", False):
            self.apply_global_randomization()


        # ✅ Keep full config as snapshot
        self.settings = self.settings.copy()
        
        
        self.re_randomizable_keys = [
            "sigma_min", "sigma_max", "start_blend", "end_blend", "sharpness",
            "early_stopping_threshold", "update_interval",
            "initial_step_size", "final_step_size",
            "initial_noise_scale", "final_noise_scale",
            "smooth_blend_factor", "step_size_factor", "noise_scale_factor", "rho"
        ]
       
        for key in self.re_randomizable_keys:
            value = self.settings.get(key)
            if value is None:
                raise KeyError(f"[KEScheduler] Missing required setting: {key}")
            setattr(self, key, value)
    
        
        
    
    def __call__(self, steps, device):
        self.steps = steps or getattr(self, "steps", None)
        if self.steps is None:
            raise ValueError("Number of steps (steps) must be provided.")
        return self.compute_sigmas(n, device)
    
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return user_config
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error loading config file: {e}")
            return {}
    
    def apply_global_randomization(self):
        """Force randomization for all eligible settings by enabling _rand flags and re-randomizing values."""

        # First pass: turn on all _rand flags if corresponding _rand_min/_rand_max exists
        for key in list(self.settings.keys()):
            if key.endswith("_rand_min") or key.endswith("_rand_max"):
                base_key = key.rsplit("_rand_", 1)[0]
                rand_flag_key = f"{base_key}_rand"
                self.settings[rand_flag_key] = True
        

        # Step 2: If global_randomize is active, re-randomize all eligible keys
       
        if self.settings.get("global_randomize", False):           
            if key not in self.settings:
                raise KeyError(f"[apply_global_randomization] Missing required key: {key}")

            default_val = self.settings[key]
            randomized_val = self.get_random_or_default(key, default_val)
            self.settings[key] = randomized_val
            setattr(self, key, randomized_val)  # 🔄 update self.X too



   
    def get_random_or_default(self, key_prefix, default_value):
        """Helper function to either randomize a value based on conditions or return the default."""
        
        # Determine if we should randomize based on global and individual flags
        randomize_flag = self.settings.get(f'{key_prefix}_rand', False)

        if randomize_flag:
            # Use specified min/max values for randomization if they exist, else use default range
            rand_min = self.settings.get(f'{key_prefix}_rand_min', default_value * 0.8)
            rand_max = self.settings.get(f'{key_prefix}_rand_max', default_value * 1.2)
            value = random.uniform(rand_min, rand_max)
            # logging.info(f"Randomized {key_prefix}: {value}")
        else:
            # Use default value if no randomization is applied
            value = default_value
            # logging.info(f"Using default {key_prefix}: {value}")

        return value 
    
    def _start_sigmas_karras(self, steps, sigma_min, sigma_max, device):
        """Retrieve randomized sigma_min and sigma_max for Karras using the existing structured randomizer."""      
        self.sigma_min = self.get_random_or_default('sigma_min', sigma_min)
        self.sigma_max = self.get_random_or_default('sigma_max', sigma_max)

        if self.sigma_min >= self.sigma_max:
            self.sigma_min = random.uniform(0.001, 0.05)  # Ensure sigma_min < sigma_max

            # logging.info(f"Karras Adjusted sigma_min={sigma_min}, sigma_max={sigma_max}")
            print(f"Debugging Karras: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")

        return self.steps, self.sigma_min, self.sigma_max, device



    
    def _start_sigmas_exponential(self, steps, sigma_min, sigma_max, device):
        """Retrieve randomized sigma_min and sigma_max for Exponential using the existing structured randomizer."""
        
        self.sigma_min = self.get_random_or_default('sigma_min', sigma_min)
        self.sigma_max = self.get_random_or_default('sigma_max', sigma_max)

        if self.sigma_min >= self.sigma_max:
            self.sigma_min = random.uniform(0.001, 0.05)

            # logging.info(f"Exponential Adjusted sigma_min={sigma_min}, sigma_max={sigma_max}")
            print(f"Debugging Exponential: sigma_min={sigma_min}, sigma_max={sigma_max}")       

        return self.steps, self.sigma_min, self.sigma_max, device
    
    def compute_sigmas(self, steps:int, device:torch.device)->torch.Tensor:           
       
       
        
        if steps is None:
            raise ValueError("Number of steps must be provided.")
        if isinstance(device, str):
            device = torch.device(device)

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
       
                
        #debug_log("Entered simple_karras_exponential_scheduler function")    
        
        
        

                  
            # Use the self.get_random_or_default function for each parameter
            #if randomize = false, then it checks for each variable for randomize, if true, then that particular option is randomized, with the others using default or config defined values."
        acceptable_keys = [
            "sigma_min", "sigma_max", "start_blend", "end_blend", "sharpness",
            "early_stopping_threshold", "update_interval", "initial_step_size",
            "final_step_size", "initial_noise_scale", "final_noise_scale",
            "smooth_blend_factor", "step_size_factor", "noise_scale_factor", "rho"
        ]

        for key in acceptable_keys:
            default_val = self.settings[key]
            value = self.get_random_or_default(key, default_val)
            setattr(self, key, value)
    
         
        # Ensure sigma_min and sigma_max are safe for log operation
        self.min_threshold = random.uniform(1e-5, 5e-5)  # A small positive value to avoid log(0)
        self.sigma_min = max(self.sigma_min, self.min_threshold)
        self.sigma_max = max(self.sigma_max, self.min_threshold)

        # Ensure sigma_min < sigma_max to prevent issues
        if self.sigma_min >= self.sigma_max:
            rand_a, rand_b = 0.01, 0.99        
            random_rand = random.uniform(rand_a, rand_b)
            old_sigma_min = sigma_min  # Store old value for debugging
            self.sigma_min = self.sigma_max * random_rand  # Ensure sigma_min is always smaller than sigma_max

            # logging.warning(
            #    f"Invalid sigma range detected! Adjusted sigma_min (was {old_sigma_min}) to {sigma_min} "
            #    f"using random factor {random_rand} to maintain sigma_min < sigma_max."
            #)
            print(
                f"Debugging Warning: sigma_min was greater than sigma_max! Adjusted it using {random_rand}. "
                f"New sigma_min={self.sigma_min}, sigma_max={self.sigma_max}"
            )

        # Now it's safe to compute sigmas
        start = math.log(self.sigma_max)
        end = math.log(self.sigma_min)
        self.sigmas = torch.linspace(start, end, self.steps, device=self.device).exp()
        #self.sigmas = torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), n, device=self.device).exp()

        # Ensure sigmas contain valid values before using them
        if torch.any(self.sigmas > 0):  
            self.sigma_min, self.sigma_max = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
            # logging.info(f"Using computed sigma values: sigma_min={sigma_min}, sigma_max={sigma_max}")
            #print(f"Debugging: Computed sigma values -> sigma_min={sigma_min}, sigma_max={sigma_max}")
        else:
            # If sigmas are all invalid, set a safe fallback
            self.sigma_min, self.sigma_max = self.min_threshold, self.min_threshold  
            # logging.warning(f"No positive sigma values found! Using fallback: sigma_min={sigma_min}, sigma_max={sigma_max}")
            print(f"Debugging Warning: No positive sigma values found! Setting fallback sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")


       
        # logging.info(f"Using device: {device}")
        # Generate sigma sequences using Karras and Exponential methods
        self._start_sigmas_exponential(steps=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device)
        self._start_sigmas_karras(steps=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device)
        
        self.sigmas_karras = get_sigmas_karras(n=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho, device=self.device)
        self.sigmas_exponential = get_sigmas_exponential(n=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device) 
        print(f"Randomized Values: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
                  
        # Match lengths of sigma sequences
        target_length = min(len(self.sigmas_karras), len(self.sigmas_exponential))  
        self.sigmas_karras = self.sigmas_karras[:target_length]
        self.sigmas_exponential = self.sigmas_exponential[:target_length]
                  
        # logging.info(f"Generated sigma sequences. Karras: {sigmas_karras}, Exponential: {sigmas_exponential}")
        
        
      
        
        if self.sigmas_karras is None:
            raise ValueError(f"Sigmas Karras:{self.sigmas_karras} Failed to generate or assign sigmas correctly.")
        if self.sigmas_exponential is None:    
            raise ValueError(f"Sigmas Exponential: {self.sigmas_exponential} Failed to generate or assign sigmas correctly.")
            #sigmas_karras = torch.zeros(steps).to(device)
            #sigmas_exponential = torch.zeros(steps).to(device)   
        try:
            pass
        except Exception as e:
            logging.warning(f"Error generating sigmas: {e}")
        
            # logging.info(f".")
        
         # Expand sigma_max slightly to account for smoother transitions  
        self.sigma_max = self.sigma_max * 1.1
      
        # Define progress and initialize blend factor
        progress = torch.linspace(0, 1, len(self.sigmas_karras)).to(self.device)
        # logging.info(f"Progress created {progress}")
        # logging.info(f"Progress Using device: {device}")
        
        sigs = torch.zeros_like(self.sigmas_karras).to(self.device)
        # logging.info(f"Sigs created {sigs}")
        # logging.info(f"Sigs Using device: {device}")

        # Iterate through each step, dynamically adjust blend factor, step size, and noise scaling
        
        if len(self.sigmas_karras) < len(self.sigmas_exponential):
            # Pad `sigmas_karras` with the last value
            padding_karras = torch.full((len(self.sigmas_exponential) - len(self.sigmas_karras),), self.sigmas_karras[-1]).to(self.sigmas_karras.self.device)
            self.sigmas_karras = torch.cat([self.sigmas_karras, padding_karras])
        elif len(self.sigmas_karras) > len(self.sigmas_exponential):
            # Pad `sigmas_exponential` with the last value
            padding_exponential = torch.full((len(self.sigmas_karras) - len(self.sigmas_exponential),), self.sigmas_exponential[-1]).to(self.sigmas_exponential.device)
            self.sigmas_exponential = torch.cat([self.sigmas_exponential, padding_exponential])
       
        for i in range(len(self.sigmas_karras)):    
                  
            # Adaptive step size and blend factor calculations
            self.step_size = self.initial_step_size * (1 - progress[i]) + self.final_step_size * progress[i] * self.step_size_factor  # 0.8 default value Adjusted to avoid over-smoothing
            # logging.info(f"Step_size created {step_size}"   )
            self.dynamic_blend_factor = self.start_blend * (1 - progress[i]) + self.end_blend * progress[i]
            # logging.info(f"Dynamic_blend_factor created {dynamic_blend_factor}"  )
            self.noise_scale = self.initial_noise_scale * (1 - progress[i]) + self.final_noise_scale * progress[i] * self.noise_scale_factor  # 0.9 default value Adjusted to keep more variation
            # logging.info(f"noise_scale created {noise_scale}"   )
            
            # Calculate smooth blending between the two sigma sequences
            self.smooth_blend = torch.sigmoid((self.dynamic_blend_factor - 0.5) * self.smooth_blend_factor) # Increase scaling factor to smooth transitions more
            # logging.info(f"smooth_blend created {smooth_blend}"   )
            
            # Compute blended sigma values
            blended_sigma = self.sigmas_karras[i] * (1 - self.smooth_blend) + self.sigmas_exponential[i] * self.smooth_blend
            # logging.info(f"blended_sigma created {blended_sigma}"   )
            
            # Apply step size and noise scaling
            sigs[i] = blended_sigma * self.step_size * self.noise_scale

        # Optional: Adaptive sharpening based on sigma values
        sharpen_mask = torch.where(sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
        # logging.info(f"sharpen_mask created {sharpen_mask} with device {self.device}"   )
        sigs = sigs * sharpen_mask
        
        # Implement early stop criteria based on sigma convergence
        change = torch.abs(sigs[1:] - sigs[:-1])
        if torch.all(change < self.early_stopping_threshold):
            # logging.info("Early stopping criteria met."   )
            return sigs[:len(change) + 1].to(self.device)
        
        if torch.isnan(sigs).any() or torch.isinf(sigs).any():
            raise ValueError("Invalid sigma values detected (NaN or Inf).")

        return sigs.to(self.device)
       


