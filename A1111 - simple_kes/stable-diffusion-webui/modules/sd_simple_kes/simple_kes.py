import torch
import logging
from modules.sd_simple_kes.get_sigmas import get_sigmas_karras, get_sigmas_exponential 
import os
import yaml
import random
from datetime import datetime
import warnings
import math
from typing import Optional
import json
from modules.sd_simple_kes.validate_config import validate_config



def simple_kes_scheduler(n: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    scheduler = SimpleKEScheduler(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    return scheduler()
    
class SharedLogger:
    def __init__(self, debug=False):
        self.debug = debug
        self.log_buffer = []

    def log(self, message):
        if self.debug:
            self.log_buffer.append(message)

    def dump(self):
        return "\n".join(self.log_buffer)

    
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
    
    
    def __init__(self, n: int, sigma_min: Optional[float] = None, sigma_max: Optional[float] = None, device: torch.device = "cuda", logger=None, **kwargs)->torch.Tensor:         
        self.steps = n if n is not None else 25         
        self.device = torch.device(device if isinstance(device, str) else device)        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        self.RANDOMIZATION_TYPE_ALIASES = {
            'symmetric': 'symmetric', 'sym': 'symmetric', 's': 'symmetric',
            'asymmetric': 'asymmetric', 'assym': 'asymmetric', 'a': 'asymmetric',
            'off': 'off', 'none': 'off'
        }  
        
        # Temporarily hold overrides from kwargs
        self._overrides = kwargs.copy()        
        self.config_path = os.path.abspath(os.path.normpath(os.path.join("modules", "sd_simple_kes", "kes_config", "default_config.yaml")))
        self.config_data = self.load_config()        
        self.config = self.config_data.copy()
        self.settings = self.config.copy() 
                
        # Apply overrides from kwargs if present
        for k, v in self._overrides.items():
            if k in self.settings:
                self.settings[k] = v
                setattr(self, k, v)        
        
        self.debug = self.settings.get("debug", False)
        logger = SharedLogger(debug=self.debug)
        self.logger=logger
        self.log = self.logger.log
        validate_config(self.config, logger=self.logger)
   
        for key, value in self.settings.items():            
            setattr(self, key, value)
            
        if self.settings.get("global_randomize", False):
            self.apply_global_randomization()
        self.settings = self.settings.copy()        
        
        self.re_randomizable_keys = [
            "sigma_min", "sigma_max", "start_blend", "end_blend", "sharpness",
            "early_stopping_threshold",
            "initial_step_size", "final_step_size",
            "initial_noise_scale", "final_noise_scale",
            "smooth_blend_factor", "step_size_factor", "noise_scale_factor", "rho"
        ]
       
        for key in self.re_randomizable_keys:
            value = self.settings.get(key)
            if value is None:
                raise KeyError(f"[KEScheduler] Missing required setting: {key}")
            setattr(self, key, value)
    
          
                     
    
    def __call__(self):        
        sigmas = self.compute_sigmas()
        if torch.isnan(sigmas).any():
            raise ValueError("[SimpleKEScheduler] NaN detected in sigmas")
        if torch.isinf(sigmas).any():
            raise ValueError("[SimpleKEScheduler] Inf detected in sigmas")
        if (sigmas <= 0).all():
            raise ValueError("[SimpleKEScheduler] All sigma values are <= 0")
        if (sigmas > 1000).all():
            raise ValueError("[SimpleKEScheduler] Sigma values are extremely large — might explode the model")

        self.save_generation_settings()
        return sigmas
      
    def save_generation_settings(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = os.path.join("modules", "sd_simple_kes", "image_generation_data")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"generation_log_{timestamp}.txt")

        with open(filename, "w") as f:
            for line in self.logger.log_buffer:  # 🔄 Use logger's buffer
                f.write(f"{line}\n")

        self.log(f"[SimpleKEScheduler] Generation log saved to {filename}")
        self.logger.log_buffer.clear()  # 🔄 Clear logger’s buffer, not self.log_buffer   
    
    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return user_config
        except FileNotFoundError:
            self.log(f"Config file not found: {self.config_path}. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            self.log(f"Error loading config file: {e}")
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
        auto_enabled = self.sigma_auto_enabled
        auto_mode = self.sigma_auto_mode

        # Handle randomization type and percent (allow per-key override)
        global_type_raw = self.settings.get('randomization_type', 'asymmetric').lower()
        randomization_type = self.RANDOMIZATION_TYPE_ALIASES.get(global_type_raw, 'asymmetric')

        key_type_raw = self.settings.get(f'{key_prefix}_randomization_type', None)
        if key_type_raw:
            key_type_resolved = self.RANDOMIZATION_TYPE_ALIASES.get(key_type_raw.lower())
            if key_type_resolved:
                randomization_type = key_type_resolved

        # Handle randomization percent
        randomization_percent = self.settings.get('randomization_percent', 0.2)
        per_key_percent = self.settings.get(f'{key_prefix}_randomization_percent', None)
        if per_key_percent is not None:
            randomization_percent = per_key_percent

        randomize_flag = self.settings.get(f'{key_prefix}_rand', False)

        if randomize_flag:
            if randomization_type == 'symmetric':
                rand_min = default_value * (1 - randomization_percent)
                rand_max = default_value * (1 + randomization_percent)
                self.log(f"[Symmetric Randomization] {key_prefix}: Range {rand_min} to {rand_max}")

            elif randomization_type == 'asymmetric':
                rand_min = default_value * (1 - randomization_percent)
                rand_max = default_value * (1 + (randomization_percent * 2))
                self.log(f"[Asymmetric Randomization] {key_prefix}: Range {rand_min} to {rand_max}")

            elif randomization_type == 'off':
                rand_min = self.settings.get(f'{key_prefix}_rand_min', default_value)
                rand_max = self.settings.get(f'{key_prefix}_rand_max', default_value)
                self.log(f"[Randomization Off] {key_prefix}: Manual range {rand_min} to {rand_max}")           

            else:
                raise ValueError(f"[Config Error] Invalid randomization_type: {randomization_type}.")

            value = random.uniform(rand_min, rand_max)
            self.log(f"Randomized {key_prefix}: {value}")

        else:
            value = default_value
            self.log(f"Using default {key_prefix}: {value}")

        # Apply auto mode override if needed        
        if auto_enabled:
            if auto_mode == 'sigma_min' and key_prefix == 'sigma_min':
                value = self.sigma_max / self.sigma_scale_factor
                self.log(f"[Auto Mode Override] sigma_min is auto-controlled. Calculated as sigma_max / scale_factor: {value}")

            if auto_mode == 'sigma_max' and key_prefix == 'sigma_max':
                value = self.sigma_min * self.sigma_scale_factor
                self.log(f"[Auto Mode Override] sigma_max is auto-controlled. Calculated as sigma_min * scale_factor: {value}")
        return value

    def start_sigmas(self, steps, sigma_min, sigma_max, device):
        """Retrieve randomized sigma_min and sigma_max using the structured randomizer, respecting auto mode."""

        #auto_enabled = self.settings.get("sigma_auto_enabled", False)
        #auto_mode = self.settings.get("sigma_auto_mode", "sigma_min")
        #scale_factor = self.settings.get("sigma_scale_factor", 1000)

        # Apply randomization, respecting auto mode
        self.sigma_min = self.get_random_or_default('sigma_min', sigma_min)
        self.sigma_max = self.get_random_or_default('sigma_max', sigma_max)

        # Apply auto scaling if enabled
        if self.sigma_auto_enabled:
            if self.sigma_auto_mode == "sigma_min":
                self.sigma_min = self.sigma_max / self.sigma_scale_factor
                self.log(f"[Auto Sigma Min] sigma_min set to {self.sigma_min} using scale factor {self.sigma_scale_factor}")

            elif self.sigma_auto_mode == "sigma_max":
                self.sigma_max = self.sigma_min * self.sigma_scale_factor
                self.log(f"[Auto Sigma Max] sigma_max set to {self.sigma_max} using scale factor {self.sigma_scale_factor}")

        # Ensure sigma_min is less than sigma_max
        if self.sigma_min >= self.sigma_max:
            correction_factor = random.uniform(0.01, 0.99)
            old_sigma_min = self.sigma_min
            self.sigma_min = self.sigma_max * correction_factor
            self.log(f"[Correction] sigma_min ({old_sigma_min}) was >= sigma_max ({self.sigma_max}). Adjusted sigma_min to {self.sigma_min} using correction factor {correction_factor}.")

        self.log(f"Final sigmas: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
        return steps, self.sigma_min, self.sigma_max, device
    
    def compute_sigmas(self)->torch.Tensor:           
       
        if self.steps is None:
            raise ValueError("Number of steps must be provided.")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

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
            initial_step_size (float): Initial step size for adaptive step size calculation.
            final_step_size (float): Final step size for adaptive step size calculation.
            initial_noise_scale (float): Initial noise scale factor.
            final_noise_scale (float): Final noise scale factor.
            step_size_factor: Adjust to compensate for oversmoothing
            noise_scale_factor: Adjust to provide more variation
            
        Returns:
            torch.Tensor: A tensor of blended sigma values.
        """
             
                  
        # Use the self.get_random_or_default function for each parameter
        #if randomize = false, then it checks for each variable for randomize, if true, then that particular option is randomized, with the others using default or config defined values."
        acceptable_keys = [
            "sigma_min", "sigma_max", "start_blend", "end_blend", "sharpness",
            "early_stopping_threshold", "initial_step_size",
            "final_step_size", "initial_noise_scale", "final_noise_scale",
            "smooth_blend_factor", "step_size_factor", "noise_scale_factor", "rho"
        ]

        for key in acceptable_keys:
            default_val = self.settings[key]
            value = self.get_random_or_default(key, default_val)
            setattr(self, key, value)    
         
        
        if self.sigma_auto_enabled:
            if self.sigma_auto_mode not in ["sigma_min", "sigma_max"]:
                raise ValueError(f"[Config Error] Invalid sigma_auto_mode: {self.sigma_auto_mode}. Must be 'sigma_min' or 'sigma_max'.")

            if self.sigma_auto_mode == "sigma_min":
                self.sigma_min = self.sigma_max / self.sigma_scale_factor
                self.log(f"[Auto Sigma Min] sigma_min set to {self.sigma_min} using scale factor {self.sigma_scale_factor}")

            elif self.sigma_auto_mode == "sigma_max":
                self.sigma_max = self.sigma_min * self.sigma_scale_factor
                self.log(f"[Auto Sigma Max] sigma_max set to {self.sigma_max} using scale factor {self.sigma_scale_factor}")

        # Always apply min_threshold AFTER auto scaling
        self.min_threshold = random.uniform(1e-5, 5e-5)

        if self.sigma_min < self.min_threshold:
            self.log(f"[Threshold Enforcement] sigma_min was too low: {self.sigma_min} < min_threshold {self.min_threshold}")
            self.sigma_min = self.min_threshold

        if self.sigma_max < self.min_threshold:
            self.log(f"[Threshold Enforcement] sigma_max was too low: {self.sigma_max} < min_threshold {self.min_threshold}")
            self.sigma_max = self.min_threshold
            
        # Now it's safe to compute sigmas
        start = math.log(self.sigma_max)
        end = math.log(self.sigma_min)
        self.sigmas = torch.linspace(start, end, self.steps, device=self.device).exp()       

        # Ensure sigmas contain valid values before using them
        if torch.any(self.sigmas > 0):  
            self.sigma_min, self.sigma_max = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()            
        else:
            # If sigmas are all invalid, set a safe fallback
            self.sigma_min, self.sigma_max = self.min_threshold, self.min_threshold              
            self.log(f"Debugging Warning: No positive sigma values found! Setting fallback sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
            
        self.log(f"Using device: {self.device}")
        # Generate sigma sequences using Karras and Exponential methods
        self.start_sigmas(steps=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device)
        self.sigmas_karras = get_sigmas_karras(n=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho, device=self.device)
        self.sigmas_exponential = get_sigmas_exponential(n=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device) 
        self.log(f"Randomized Values: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
                  
        # Match lengths of sigma sequences
        target_length = min(len(self.sigmas_karras), len(self.sigmas_exponential))  
        self.sigmas_karras = self.sigmas_karras[:target_length]
        self.sigmas_exponential = self.sigmas_exponential[:target_length]
                  
        self.log(f"Generated sigma sequences. Karras: {self.sigmas_karras}, Exponential: {self.sigmas_exponential}")
        
        if self.sigmas_karras is None:
            raise ValueError(f"Sigmas Karras:{self.sigmas_karras} Failed to generate or assign sigmas correctly.")
        if self.sigmas_exponential is None:    
            raise ValueError(f"Sigmas Exponential: {self.sigmas_exponential} Failed to generate or assign sigmas correctly.")
            self.sigmas_karras = torch.zeros(self.steps).to(self.device)
            self.sigmas_exponential = torch.zeros(self.steps).to(self.device)   
        try:
            pass
        except Exception as e:
            self.log(f"Error generating sigmas: {e}")
        
         # Expand sigma_max slightly to account for smoother transitions  
        self.sigma_max = self.sigma_max * 1.1
      
        # Define progress and initialize blend factor
        self.progress = torch.linspace(0, 1, len(self.sigmas_karras)).to(self.device)
        self.log(f"Progress created {self.progress}")
        self.log(f"Progress Using device: {self.device}")
        
        sigs = torch.zeros_like(self.sigmas_karras).to(self.device)
        self.log(f"Sigs created {sigs}")        

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
            self.step_size = self.initial_step_size * (1 - self.progress[i]) + self.final_step_size * self.progress[i] * self.step_size_factor  # 0.8 default value Adjusted to avoid over-smoothing
            
            self.dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]            
            self.noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor  
            smooth_blend = torch.sigmoid((self.dynamic_blend_factor - 0.5) * self.smooth_blend_factor) # Increase scaling factor to smooth transitions more           
            
            # Compute blended sigma values
            blended_sigma = self.sigmas_karras[i] * (1 - smooth_blend) + self.sigmas_exponential[i] * smooth_blend           
            
            # Apply step size and noise scaling
            sigs[i] = blended_sigma * self.step_size * self.noise_scale

        # Optional: Adaptive sharpening based on sigma values
        self.sharpen_mask = torch.where(sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
        self.log(f"sharpen_mask created {self.sharpen_mask} with device {self.device}"   )
        sigs = sigs * self.sharpen_mask
        #self.log(f"Sigs after sharpen_mask: {sigs}")
        
        # Implement early stop criteria based on sigma convergence
        change = torch.abs(sigs[1:] - sigs[:-1])
        if torch.all(change < self.early_stopping_threshold):
            self.log("Early stopping criteria met."   )
            return sigs[:len(change) + 1].to(self.device)
        
        if torch.isnan(sigs).any() or torch.isinf(sigs).any():
            raise ValueError("Invalid sigma values detected (NaN or Inf).")

        return sigs.to(self.device)
