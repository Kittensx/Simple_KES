import torch
import logging
from modules.sd_simple_kes.get_sigmas import get_sigmas_karras, get_sigmas_exponential 
from modules.sd_simple_kes.validate_config import validate_config
from modules.sd_simple_kes.plot_sigma_sequence import plot_sigma_sequence
import os
import yaml
import random
from datetime import datetime
import warnings
import math
from typing import Optional
import json
import numpy as np



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
            'logarithmic': 'logarithmic', 'log': 'logarithmic', 'l': 'logarithmic',
            'exponential': 'exponential', 'exp': 'exponential', 'e': 'exponential'
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
        
        self.initialize_generation_filename()
    

        
    def __call__(self):
        # First pass: Run prepass to determine predicted_stop_step
        if not self.settings.get('skip_prepass', False):
            final_steps = self.prepass_compute_sigmas()  
            sigmas = self.compute_sigmas(final_steps)
            
            
            
        else:
           # Build sigma sequence directly (without prepass)
            #self.config_values()
            #self.start_sigmas(sigma_min=self.sigma_min, sigma_max=self.sigma_max)
            #self.generate_sigmas_schedule()            
            
            self.blend_sigma_sequence(
                sigs=self.sigs,
                sigmas_karras=self.sigmas_karras,
                sigmas_exponential=self.sigmas_exponential,
                pre_pass = False
                
            )
            final_steps = n if n is not none else self.steps
        
            sigmas = self.compute_sigmas(final_steps)
       
        
        # Safety checks
        if torch.isnan(sigmas).any():
            raise ValueError("[SimpleKEScheduler] NaN detected in sigmas")
        if torch.isinf(sigmas).any():
            raise ValueError("[SimpleKEScheduler] Inf detected in sigmas")
        if (sigmas <= 0).all():
            raise ValueError("[SimpleKEScheduler] All sigma values are <= 0")
        if (sigmas > 1000).all():
            raise ValueError("[SimpleKEScheduler] Sigma values are extremely large — might explode the model")

        # Save logs to file
        self.save_generation_settings()
        
        
        # Return final sigmas to the scheduler caller
        return sigmas

    def initialize_generation_filename(self, folder=None, base_name="generation_log", ext="txt"):
        """
        Initialize the log filename early so it can be used throughout the process.
        """
        if folder is None:
            folder = self.settings.get('log_save_directory', 'modules/sd_simple_kes/image_generation_data')
            folder = os.path.abspath(os.path.normpath(folder))

        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_filename = os.path.join(folder, f"{base_name}_{timestamp}.{ext}")
        #return self.log_filename

      
    def save_generation_settings(self):
        """
        Save the generation log with configurable directory, base name, and extension.

        Parameters:
        - folder (str): Optional custom directory to save the log file.
        - base_name (str): The base name for the file (default is 'generation_log').
        - ext (str): The file extension to use (default is 'txt').
        """        

        with open(self.log_filename, "w", encoding = 'utf-8') as f:
            for line in self.logger.log_buffer:
                f.write(f"{line}\n")

        self.log(f"[SimpleKEScheduler] Generation log saved to {self.log_filename}")
        self.logger.log_buffer.clear() 
    
    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding = 'utf-8') as f:
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
            setattr(self, key, randomized_val)  
   
    def get_randomization_type(self, key_prefix):
        """
        Retrieves the randomization type for a given key, with fallback to 'asymmetric' if missing.
        """
        randomization_type_raw = self.settings.get(f'{key_prefix}_randomization_type', 'asymmetric')
        randomization_type = self.RANDOMIZATION_TYPE_ALIASES.get(randomization_type_raw.lower(), 'asymmetric')
        return randomization_type

    def get_randomization_percent(self, key_prefix):
        """
        Retrieves the randomization percent for a given key, with fallback to 0.2 if missing.
        """
        return self.settings.get(f'{key_prefix}_randomization_percent', 0.2)

   
    def get_random_between_min_max(self, key_prefix, default_value):
        """
        Picks a random value between _rand_min and _rand_max if _rand is True.
        Otherwise, returns the base value.
        """
        randomize_flag = self.settings.get(f'{key_prefix}_rand', False)

        if randomize_flag:
            rand_min = self.settings.get(f'{key_prefix}_rand_min', default_value)
            rand_max = self.settings.get(f'{key_prefix}_rand_max', default_value)

            if rand_min == rand_max:
                self.log(f"[Random Range] {key_prefix}: min and max are equal ({rand_min}). Using single value.")
                return rand_min

            value = random.uniform(rand_min, rand_max)
            self.log(f"[Random Range] {key_prefix}: Picked random value {value} between {rand_min} and {rand_max}")
            return value
        else:
            self.log(f"[Random Range] {key_prefix}: Randomization is OFF. Using base value {default_value}")
            return default_value

    def get_random_by_type(self, key_prefix, default_value):
        randomization_enabled = self.settings.get(f'{key_prefix}_enable_randomization_type', False)

        if not randomization_enabled:
            self.log(f"[Randomization Type] {key_prefix}: Randomization type is OFF. Using base value {default_value}")
            return default_value

        randomization_type = self.get_randomization_type(key_prefix)
        randomization_percent = self.get_randomization_percent(key_prefix)

        if randomization_type == 'symmetric':
            rand_min = default_value * (1 - randomization_percent)
            rand_max = default_value * (1 + randomization_percent)
            self.log(f"[Symmetric Randomization] {key_prefix}: Range {rand_min} to {rand_max}")
            

        elif randomization_type == 'asymmetric':
            rand_min = default_value * (1 - randomization_percent)
            rand_max = default_value * (1 + (randomization_percent * 2))
            self.log(f"[Asymmetric Randomization] {key_prefix}: Range {rand_min} to {rand_max}")

        elif randomization_type == 'logarithmic':
            rand_min = math.log(default_value * (1 - randomization_percent))
            rand_max = math.log(default_value * (1 + randomization_percent))
            value = math.exp(random.uniform(rand_min, rand_max))
            self.log(f"[Logarithmic Randomization] {key_prefix}: Log-space randomization resulted in {value}")
            return value

        elif randomization_type == 'exponential':
            rand_min = default_value * (1 - randomization_percent)
            rand_max = default_value * (1 + randomization_percent)
            base_value = random.uniform(rand_min, rand_max)
            value = math.exp(base_value)
            self.log(f"[Exponential Randomization] {key_prefix}: Randomized exponential value {value}")
            return value

        else:
            self.log(f"[Randomization Type] {key_prefix}: Invalid randomization type {randomization_type}. Using base value.")
            return default_value

        value = random.uniform(rand_min, rand_max)
        
        self.log(f"[Randomization Type] {key_prefix}: Randomized value {value}")
        return value

    def get_random_or_default(self, key_prefix, default_value):
        """
        Selects randomization method based on active flags:
            - If both enabled → prioritize randomization type (or min/max if you prefer).
            - If only one enabled → apply that one.
            - If neither → return default value.
        """
        rand_type_enabled = self.settings.get(f'{key_prefix}_enable_randomization_type', False)
        min_max_enabled = self.settings.get(f'{key_prefix}_rand', False)

        if rand_type_enabled and min_max_enabled:
            self.log(f"[Randomization Policy] Both min/max and randomization type enabled for {key_prefix}. System will prioritize randomization type.")
            result_value = self.get_random_by_type(key_prefix, default_value)

        elif rand_type_enabled:
            result_value = self.get_random_by_type(key_prefix, default_value)
            self.log(f"[Randomization] {key_prefix}: Applied randomization type. Final value: {result_value}")

        elif min_max_enabled:
            result_value = self.get_random_between_min_max(key_prefix, default_value)
            self.log(f"[Randomization] {key_prefix}: Applied min/max randomization. Final value: {result_value}")

        else:
            result_value = default_value
            self.log(f"[Randomization] {key_prefix}: No randomization applied. Using default value: {result_value}")

        return result_value

    def start_sigmas(self, sigma_min, sigma_max):
        """Ensures sigma_min is always less than sigma_max for edge cases"""
       
        if sigma_min >= sigma_max:
            correction_factor = random.uniform(0.01, 0.99)
            old_sigma_min = sigma_min
            sigma_min = sigma_max * correction_factor
            self.log(f"[Correction] sigma_min ({old_sigma_min}) was >= sigma_max ({sigma_max}). Adjusted sigma_min to {sigma_min} using correction factor {correction_factor}.")

        self.log(f"Final sigmas: sigma_min={sigma_min}, sigma_max={sigma_max}")
        return sigma_min, sigma_max
    


    def blend_sigma_sequence(self, sigs, sigmas_karras, sigmas_exponential, pre_pass=False):
        self.progress = torch.linspace(0, 1, len(sigmas_karras)).to(self.device)
        self.blended_sigmas = []
        self.change_log = []
        
        # === Final Pass ===
        self.log("\n--- Starting Final Pass Blending ---\n")
        for i in range(len(sigmas_karras)):
            if self.step_progress_mode == "linear":
                progress_value = self.progress[i]
            elif self.step_progress_mode == "exponential":
                progress_value = self.progress[i] ** self.settings.get("exp_power", 2)
            elif self.step_progress_mode == "logarithmic":
                progress_value = torch.log1p(self.progress[i] * (torch.exp(torch.tensor(1.0)) - 1))
            elif self.step_progress_mode == "sigmoid":
                progress_value = 1 / (1 + torch.exp(-12 * (self.progress[i] - 0.5)))
            else:
                progress_value = self.progress[i]  # Fallback to linear (previous version used)            
            self.step_size = self.initial_step_size * (1 - progress_value) + self.final_step_size * progress_value * self.step_size_factor
            if i == 0:
                step_label = "First Step"
            elif i == len(sigmas_karras) // 2:
                step_label = "Middle Step"
            elif i == len(sigmas_karras) - 1:
                step_label = "Last Step"
            else:
                step_label = None
            
            dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]
            smooth_blend = torch.sigmoid((dynamic_blend_factor - 0.5) * self.smooth_blend_factor)            
            noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor                        
            blended_sigma = sigmas_karras[i] * (1 - smooth_blend) + sigmas_exponential[i] * smooth_blend
            

            sigs[i] = blended_sigma * self.step_size * noise_scale
            self.blended_sigmas.append(blended_sigma.item())
            
            if step_label:
                if not pre_pass:  # Only log detailed steps in the final pass
                    self.log("\n" + "=" * 10 + "\n[Start of Sigma Sequence Logging]\n" + "=" * 10)
                    self.log(f"[{step_label} - Step {i+1}/{len(sigmas_karras)}]"
                             f"\nStep Size: {self.step_size:.6f}"
                             f"\nDynamic Blend Factor: {dynamic_blend_factor:.6f}"
                             f"\nNoise Scale: {noise_scale:.6f}"
                             f"\nSmooth Blend: {smooth_blend:.6f}"
                             f"\nBlended Sigma: {blended_sigma:.6f}"
                             f"\nFinal Sigma: {sigs[i]:.6f}")
                    self.log("\n" + "=" * 10 + "\n[End of Sigma Sequence Logging]\n" + "=" * 10)

                elif pre_pass:  # Optional: Log a simple summary in the prepass
                    self.log(f"[Prepass {step_label} - Step {i+1}/{len(sigmas_karras)}] "
                             f"Blended Sigma: {blended_sigma:.6f}, Final Sigma: {sigs[i]:.6f}")
            
            if i == 0:
                step_label = "First Step"
            elif i == len(sigmas_karras) - 1:
                step_label = "Last Step"
            else:
                step_label = None

            if step_label:
                self.log("\n" + "=" * 10 + "\n[Start of Sigma Sequence Logging]\n" + "=" * 10)
                self.log(f"[{step_label} - Step {i+1}/{len(sigmas_karras)}]"
                         f"\nStep Size: {self.step_size:.6f}"
                         f"\nBlended Sigma: {blended_sigma:.6f}"
                         f"\nFinal Sigma: {sigs[i]:.6f}")
                self.log("\n" + "=" * 10 + "\n[End of Sigma Sequence Logging]\n" + "=" * 10)

            if i > 0:
                self.change = torch.abs(sigs[i] - sigs[i - 1])
                self.change_log.append(self.change.item())

            # Early Stopping Evaluation
            if i > self.safety_minimum_stop_step and len(self.change_log) > 5:
                #relative_sigma_progress = (blended_sigma - self.sigs[-1].item()) / blended_sigma                
                final_target_sigma = sigmas_karras[-1].item()  # or use min(self.sigmas) if preferred
                #relative_sigma_progress = (blended_sigma - final_target_sigma) / blended_sigma  
                if blended_sigma != 0:
                    relative_sigma_progress = (blended_sigma - final_target_sigma) / blended_sigma
                else:
                    relative_sigma_progress = 0  # Assume fully converged if blended_sigma is 0                
                

                # Optional: Show variance but no need to stop on it
                self.sigma_variance = torch.var(sigs).item() if self.device != 'cpu' else np.var(self.blended_sigmas)
                self.log(f"Sigma Variance: {self.sigma_variance:.6f}")
                
                
                
        
        if pre_pass:
            self.log("\n--- Starting Pre-Pass Blending ---\n")
            for i in range(len(sigmas_karras)):                   
                dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]
                smooth_blend = torch.sigmoid((dynamic_blend_factor - 0.5) * self.smooth_blend_factor)
                noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor                
                self.blended_sigma = sigmas_karras[i] * (1 - smooth_blend) + sigmas_exponential[i] * smooth_blend
                sigs[i] = self.blended_sigma * self.step_size * noise_scale
                self.blended_sigmas.append(self.blended_sigma.item())

                if i >= 2:
                    sigma_rate = abs(self.blended_sigmas[i] - self.blended_sigmas[i - 1])
                    previous_sigma_rate = abs(self.blended_sigmas[i - 1] - self.blended_sigmas[i - 2])
                    if sigma_rate > previous_sigma_rate:
                        self.log(f"Sigma decline is slowing down → possible plateau at step {i+1}.")

                if i == 0:
                    step_label = "Prepass First Step"
                elif i == len(sigmas_karras) - 1:
                    step_label = "Prepass Last Step"
                else:
                    step_label = None

                if step_label:
                    self.log(f"[{step_label} - Step {i+1}/{len(sigmas_karras)}] Blended Sigma: {self.blended_sigma:.6f}, Final Sigma: {sigs[i]:.6f}")
                if i >= 2:
                        sigma_rate = abs(self.blended_sigmas[i] - self.blended_sigmas[i - 1])
                        previous_sigma_rate = abs(self.blended_sigmas[i - 1] - self.blended_sigmas[i - 2])

                        if sigma_rate > previous_sigma_rate:
                            self.log(f"Sigma decline is slowing down → possible plateau.")
                if i > 0:
                    self.change = torch.abs(sigs[i] - sigs[i - 1])
                    self.change_log.append(self.change.item())
                    relative_sigma_progress = (self.blended_sigma - sigs[-1].item()) / self.blended_sigma                                
                    recent_changes = torch.abs(torch.tensor(self.change_log[-5:]))
                    max_change = torch.max(recent_changes).item()                        
                    mean_change = torch.mean(recent_changes).item()
                    percent_of_threshold = (max_change / self.early_stopping_threshold) * 100                                        
                    delta_change = abs(max_change - mean_change)

                    
                    # Check 1: Relative sigma progress
                    relative_converged = relative_sigma_progress < 0.05
                    # Check 2: Max recent sigma change
                    max_converged = max_change < self.early_stopping_threshold
                    # Check 3: Max-mean difference converged
                    delta_converged = delta_change < self.settings.get('recent_change_convergence_delta', 0.02)

                    
                # Start checking for early stopping after minimum steps
                if i > self.safety_minimum_stop_step and len(self.change_log) > 10:                  
                    # Calculate variance and dynamic threshold
                    self.blended_tensor = torch.tensor(self.blended_sigmas) 
                    if self.device == 'cpu':
                        self.sigma_variance = np.var(self.blended_sigmas)
                    else: 
                        self.sigma_variance = torch.var(sigs).item()


                    self.min_sigma_threshold = self.sigma_variance * self.settings.get('sigma_variance_scale', 0.05)  # scale factor can be tuned

                    self.log(f"\n--- Early Stopping Evaluation at Step {i+1} ---")
                    self.log(f"Current Blended Sigma: {self.blended_sigma:.6f}")
                    self.log(f"Sigma Variance: {self.sigma_variance:.6f}")
                    self.log(f"Min Sigma Threshold: {self.min_sigma_threshold:.6f} (Scale: {self.settings.get('sigma_variance_scale', 0.05)})")
                    self.log(f"Visual Sigma: {self.visual_sigma:.6f} (Config: min_visual_sigma={self.settings.get('min_visual_sigma', 10)})")
                    self.log(f"Early Stopping Threshold: {self.early_stopping_threshold:.6f} (Config Method: {self.early_stopping_method})")
                    self.log(f"Safety Minimum Stop Step: {self.safety_minimum_stop_step}")
                    self.log(f"Config: early_stopping_threshold_rand: {self.settings.get('early_stopping_threshold_rand', False)}, early_stopping_threshold_rand_min: {self.settings.get('early_stopping_threshold_rand_min', 0.001)}, early_stopping_threshold_rand_max: {self.settings.get('early_stopping_threshold_rand_max', 0.02)}")

                    # Reason for continuing (sigma still too high)
                    if self.blended_sigma > self.min_sigma_threshold:
                        self.log(f"Blended Sigma {self.blended_sigma:.6f} exceeds min sigma threshold {self.min_sigma_threshold:.6f} → Continuing.\n")
                        
                    
                    # Start Early Stopping Checks
                    
                    if self.early_stopping_method == "mean":
                        mean_change = sum(self.change_log) / len(self.change_log)
                        if mean_change < self.early_stopping_threshold:
                            skipped_steps = len(sigmas_karras) - (i + 1)
                            self.log(f"Early stopping triggered by mean at step {i}. Mean change: {mean_change:.6f}. Steps used: {i + 1}/{len(sigmas_karras)}, steps skipped: {skipped_steps}")  

                    elif self.early_stopping_method == "max":
                        #max_change = max(self.change_log)
                        if max_change < self.early_stopping_threshold:
                            skipped_steps = len(sigmas_karras) - (i + 1)
                            self.log(f"Early stopping triggered by mean at step {i}. Mean change: {max_change:.6f}. Steps used: {i + 1}/{len(sigmas_karras)}, steps skipped: {skipped_steps}")  
                            
                    elif self.early_stopping_method == "sum":
                        stable_steps = sum(
                            1 for j in range(1, len(self.change_log))
                            if abs(self.change_log[j]) < self.early_stopping_threshold * abs(sigs[j])
                        )

                        if stable_steps >= 0.8 * len(self.change_log):
                            skipped_steps = len(sigmas_karras) - (i + 1)
                            self.log(f"Early stopping triggered by sum at step {i}. Stable steps: {stable_steps}/{len(self.change_log)}. Steps used: {i + 1}/{len(sigmas_karras)}, steps skipped: {skipped_steps}")
              
                                   
                   
                    if relative_converged and max_converged and delta_converged:     
                        self.log(f"\n--- Early Stopping Evaluation at Step {i+1} ---")
                        self.log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                        self.log(f"Max Recent Sigma Change: {max_change:.6f}")
                        self.log(f"Mean Recent Sigma Change: {mean_change:.6f}")
                        self.log(f"Delta Change: {delta_change:.6f} (Target: {self.settings.get('recent_change_convergence_delta', 0.02)})")                        
                        self.log(f"Early stopping criteria met at step {i+1} based on all convergence checks.")
                        self.predicted_stop_step = i

                        if self.settings.get('graph_save_enable', False):                        
                            graph_plot = plot_sigma_sequence(
                                sigs[:i + 1],
                                i,
                                self.log_filename,
                                self.graph_save_directory,
                                self.settings.get('graph_save_enable', False)
                            )
                            self.log(f"Sigma sequence plot saved to {graph_plot}")
                        break                       
                                             

            return sigs[:self.predicted_stop_step + 1]  # Return only the usable sequence


        return sigs

    '''
    def blend_sigma_sequence(self, sigs, sigmas_karras, sigmas_exponential, pre_pass = False):
        """
        Computes the blended sigma sequence using adaptive step sizes, dynamic blend factors, 
        and noise scaling across the progress of the diffusion process.

        This method blends sigma values from the Karras and Exponential schedules using 
        a smooth, progress-dependent interpolation. It applies adaptive scaling based on 
        step size and noise scale factors to each sigma in the sequence.

        Parameters:
        -----------
        sigs : torch.Tensor
            A pre-allocated tensor where the computed sigma sequence will be stored. 
            This tensor must match the shape of the sigma schedules.
        
        sigmas_karras : torch.Tensor
            The sigma sequence generated using the Karras schedule.
        
        sigmas_exponential : torch.Tensor
            The sigma sequence generated using the Exponential schedule.

        Returns:
        --------
        sigs : torch.Tensor
            The final blended and scaled sigma sequence.

        Notes:
        ------
        - This method is used in both the prepass and final pass of the scheduler.
        - The progress tensor is computed linearly from 0 to 1 over the length of the sequence.
        - The method uses class attributes for step size factors, blend factors, and noise scaling.
        - This method modifies `sigs` in place.
        """        
    '''
    def generate_sigmas_schedule(self):
        """
        Generates the sigma schedules required for the hybrid blending process.

        The Karras and Exponential sigma sequences are created to provide two distinct 
        noise scaling strategies:
        - The Karras sequence offers a more aggressive noise decay, commonly used in 
          modern schedulers for improved image quality and denoising stability.
        - The Exponential sequence provides a traditional log-space noise schedule.

        These two sequences are dynamically blended in later steps using progress-dependent 
        weights to produce a custom sigma path that combines the advantages of both approaches.

        This blending process is critical to the scheduler's ability to:
        - Adapt noise scaling across steps.
        - Control the sharpness and smoothness of transitions.
        - Support early stopping based on sigma convergence patterns.

        These sigma sequences must be regenerated in both the prepass (for early stopping detection) 
        and the final pass (for polished sigma application), ensuring both passes are synchronized 
        with the current step count and randomization settings.
        """

        
        self.sigmas_karras = get_sigmas_karras(n=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho, device=self.device)[:self.steps]
        self.sigmas_exponential = get_sigmas_exponential(n=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, device=self.device)
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
            
        if len(self.sigmas_karras) < len(self.sigmas_exponential):
            # Pad `sigmas_karras` with the last value
            padding_karras = torch.full((len(self.sigmas_exponential) - len(self.sigmas_karras),), self.sigmas_karras[-1]).to(self.sigmas_karras.self.device)
            self.sigmas_karras = torch.cat([self.sigmas_karras, padding_karras])
        elif len(self.sigmas_karras) > len(self.sigmas_exponential):
            # Pad `sigmas_exponential` with the last value
            padding_exponential = torch.full((len(self.sigmas_karras) - len(self.sigmas_exponential),), self.sigmas_exponential[-1]).to(self.sigmas_exponential.device)
            self.sigmas_exponential = torch.cat([self.sigmas_exponential, padding_exponential])
 
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
         
    
    def config_values(self):        
            
        self.sharpen_mode = self.settings.get('sharpen_mode', 'full')
        
        if self.sigma_auto_enabled:
            if self.sigma_auto_mode not in ["sigma_min", "sigma_max"]:
                raise ValueError(f"[Config Error] Invalid sigma_auto_mode: {self.sigma_auto_mode}. Must be 'sigma_min' or 'sigma_max'.")

            if self.sigma_auto_mode == "sigma_min":
                self.sigma_min = self.sigma_max / self.sigma_scale_factor
                self.log(f"[Auto Sigma Min] sigma_min set to {self.sigma_min} using scale factor {self.sigma_scale_factor}")

            elif self.sigma_auto_mode == "sigma_max":
                self.sigma_max = self.sigma_min * self.sigma_scale_factor
                self.log(f"[Auto Sigma Max] sigma_max set to {self.sigma_max} using scale factor {self.sigma_scale_factor} and using a multiplier of {sigma_max_multipier} to account for smoother transitions")

        # Always apply min_threshold AFTER auto scaling
        self.min_threshold = random.uniform(1e-5, 5e-5)

        if self.sigma_min < self.min_threshold:
            self.log(f"[Threshold Enforcement] sigma_min was too low: {self.sigma_min} < min_threshold {self.min_threshold}")
            self.sigma_min = self.min_threshold

        if self.sigma_max < self.min_threshold:
            self.log(f"[Threshold Enforcement] sigma_max was too low: {self.sigma_max} < min_threshold {self.min_threshold}")
            self.sigma_max = self.min_threshold
        
        self.early_stopping_method = self.settings.get("early_stopping_method", "mean")
        valid_methods = ['mean', 'max', 'sum']
        if self.early_stopping_method not in valid_methods:
            self.log(f"[Config Correction] Invalid early_stopping_method: {self.early_stopping_method}. Defaulting to 'mean'.")
            self.early_stopping_method = 'mean'
       
        
            
    
    def prepass_compute_sigmas(self)->torch.Tensor:           
       
        if self.steps is None:
            raise ValueError("Number of steps must be provided.")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

        self.config_values()
        self.start_sigmas(sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        self.generate_sigmas_schedule()
        self.sigs = torch.zeros_like(self.sigmas_karras).to(self.device) 
        
        
        self.predicted_stop_step = self.steps - 1  # Default to full length if not stopped

        # Reset tracking variables
        self.change_log = []
       
       

        self.sigma_variance_threshold = self.settings.get('sharpen_variance_threshold', 0.01)
         
        self.N = self.settings.get('sharpen_last_n_steps', 10)
        if self.N > len(self.sigs):
            self.N = len(self.sigs)
            self.log(f"[Sharpening Notice] Requested last {self.N} steps exceeds sequence length. Using entire sequence instead.")
        
       
        self.min_visual_sigma = self.settings.get('min_visual_sigma', 10)
        self.visual_sigma = max(0.8, self.sigma_min * self.min_visual_sigma)
        self.safety_minimum_stop_step = self.settings.get('safety_minimum_stop_step', 10)
        
                
        self.blend_sigma_sequence(sigs = self.sigs, sigmas_karras=self.sigmas_karras, sigmas_exponential=self.sigmas_exponential, pre_pass = True)
        self.sigs = torch.zeros_like(self.sigmas_karras).to(self.device) 
        
        
        

        if torch.isnan(self.sigs).any() or torch.isinf(self.sigs).any():
            raise ValueError("Invalid sigma values detected (NaN or Inf).")

        final_steps = self.sigs[:self.predicted_stop_step + 1].to(self.device)
        
        return final_steps
    
     
            
        
    def compute_sigmas(self, final_steps)->torch.Tensor:  
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
            
        self.log(f"Using device: {self.device}")
        
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
         
        self.steps = len(final_steps)       
        
        self.config_values()
        self.start_sigmas(sigma_min=self.sigma_min, sigma_max=self.sigma_max)
        self.generate_sigmas_schedule()        
        
        self.sigs = torch.zeros_like(self.sigmas_karras).to(self.device) 
        self.blend_sigma_sequence(sigs=self.sigs, sigmas_karras=self.sigmas_karras, sigmas_exponential=self.sigmas_exponential, pre_pass = False)
        '''
        if self.device == 'cpu':
            self.sigma_variance = np.var(self.blended_sigmas)
            #self.sigma_variance = np.var(self.sigs.cpu().numpy())
        else: 
            self.sigma_variance = torch.var(self.sigs).item()
        '''
        self.sigma_variance = torch.var(self.sigs).item()
        
        #self.log(f"[Sigma Variance] Calculated variance in final pass: {self.sigma_variance:.6f}")
        
        if self.sharpen_mode in ['last_n', 'both']:
            if self.sigma_variance < self.sigma_variance_threshold:
                # Apply full sharpening
                self.sharpen_mask = torch.where(self.sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
                sharpen_indices = torch.where(self.sharpen_mask < 1.0)[0].tolist()
                self.sigs = self.sigs * self.sharpen_mask
                self.log(f"[Sharpen Mask] Full sharpening applied (low variance). Steps: {sharpen_indices}")

            else:
                # Apply sharpening only to the last N steps
                recent_sigs = self.sigs[-self.N:]
                sharpen_mask = torch.where(recent_sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
                sharpen_indices = torch.where(sharpen_mask < 1.0)[0].tolist()
                self.sigs[-self.N:] = recent_sigs * sharpen_mask

                # Now loop per step if desired (safely inside this block)
                for j in range(len(self.sigs) - self.N, len(self.sigs)):
                    if self.sigs[j] < self.sigma_min * 1.5:
                        old_value = self.sigs[j].item()
                        self.sigs[j] = self.sigs[j] * self.sharpness
                        self.log(f"[Sharpening] Step {j+1}: Applied sharpening. Sigma changed from {old_value:.6f} to {self.sigs[j].item():.6f}")
                    else:
                        self.log(f"[Sharpening] Step {j+1}: No sharpening applied. Sigma: {self.sigs[j].item():.6f}")

        if self.sharpen_mode in ['full', 'both']:
            # Optional: Additional full sharpening (if needed)
            self.sharpen_mask = torch.where(self.sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
            sharpen_indices = torch.where(self.sharpen_mask < 1.0)[0].tolist()
            self.sigs = self.sigs * self.sharpen_mask
            self.log(f"[Sharpen Mask] Full sharpening applied at steps: {sharpen_indices}")

        
        

        return self.sigs.to(self.device)
