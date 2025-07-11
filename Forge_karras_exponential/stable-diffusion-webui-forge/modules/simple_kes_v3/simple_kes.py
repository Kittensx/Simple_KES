from modules.sd_simple_kes.get_sigmas import scheduler_registry
from modules.sd_simple_kes.validate_config import validate_config
from modules.sd_simple_kes.plot_sigma_sequence import plot_sigma_sequence
import torch
import torch.nn.functional as F
import logging
import os
import yaml
import random
from datetime import datetime
import warnings
import math
from typing import Optional
import json
import numpy as np
import hashlib
import glob
import re
import inspect
import copy


def simple_kes_scheduler(n: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    scheduler = SimpleKEScheduler(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    return scheduler()

class SharedLogger:
    def __init__(self, debug=False):
        self.debug = debug
        self.log_buffer = []
        self.prepass_log_buffer=[]

    def log(self, message):
        if self.debug:
            self.log_buffer.append(message)
    def prepass_log(self, message):
        if self.debug:
            self.prepass_log_buffer.append(message)
   
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
    
    def __init__(self, n: int, sigma_min: Optional[float] = None, sigma_max: Optional[float] = None, device: torch.device = "cpu", logger=None, **kwargs)->torch.Tensor:   
        self.steps = n if n is not None else 10 
        self.original_steps = n
        self.device = torch.device(device if isinstance(device, str) else device)        
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.scheduler_registry = scheduler_registry  
        self.RANDOMIZATION_TYPE_ALIASES = {
            'symmetric': 'symmetric', 'sym': 'symmetric', 's': 'symmetric',
            'asymmetric': 'asymmetric', 'assym': 'asymmetric', 'a': 'asymmetric',
            'logarithmic': 'logarithmic', 'log': 'logarithmic', 'l': 'logarithmic',
            'exponential': 'exponential', 'exp': 'exponential', 'e': 'exponential'
        }
        self._config_schema = {
            'min_visual_sigma': (int, 10),
            'safety_minimum_stop_step': (int, 10),
            'auto_tail_smoothing': (bool, False),
            'auto_stabilization_sequence': (list, [
                'smooth_interpolation', 'append_tail', 'blend_tail', 'apply_decay', 'progressive_decay'
            ]),            
            'sharpen_variance_threshold': (float, 0.01),
            'sharpen_last_n_steps': (int, 10),
            'decay_pattern': (str, 'zero'),
            'sigma_save_subfolder': (str, 'saved_sigmas'),
            'load_sigma_cache': (bool, False),
            'save_sigma_cache': (bool, False),
            'graph_save_directory': (str, 'modules/sd_simple_kes/image_generation_data'),
            'graph_save_enable': (bool, False),
            'exp_power': (int, 2),
            'recent_change_convergence_delta': (float, 0.02),
            'sigma_variance_scale': (float, 0.05),
            'allow_step_expansion': (bool, False),
            'sharpen_mode': (str, 'full'),
            'blend_midpoint': (float, 0.5),
            'early_stopping_method': (str, 'mean'),
            'save_prepass_sigmas': (bool, False),
            'global_randomize': (bool, False),
            'skip_prepass': (bool, False),
            'load_prepass_sigmas': (bool, False)
        }               
        self._overrides = kwargs.copy()  #Temporarily hold overrides from kwargs
        default_config_path = os.path.abspath(os.path.normpath(os.path.join("modules", "sd_simple_kes", "kes_config", "default_config.yaml")))
        self.default_config = self._load_config(default_config_path)
        user_config_path = os.path.abspath(os.path.normpath(os.path.join("modules", "sd_simple_kes", "kes_config", "user_config.yaml")))        
        self.user_config = self._load_config(user_config_path)
        self.config_data = {**self.default_config, **self.user_config} 
        self.config = self.config_data.copy()
        self.settings = self.config.copy() 
        for key, value in self.settings.items():            
            setattr(self, key, value)
        if self.global_randomize:
            self.apply_global_randomization()  
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
        self.debug = self.settings.get('debug', False)
        #setup logging  
        logger = SharedLogger(debug=kwargs.get('debug', False)) 
        self.logger=logger
        self.log = self.logger.log
        self.prepass_log = self.logger.prepass_log 
        self._validate_config_types() 
        validate_config(self.config, logger=self.logger)
        # Apply overrides from kwargs if present
        for k, v in self._overrides.items():
            if k in self.settings:
                self.settings[k] = v
                setattr(self, k, v)   
        self.auto_mode_enabled = self.settings.get('auto_tail_smoothing', False) 
        self.initialize_generation_filename()
        self.relative_converged = False
        self.max_converged = False
        self.delta_converged = False
        self.early_stop_triggered = False
        self.sigma_cache = {}        
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(self.BASE_DIR, 'cache')
        self.sigma_save_folder = os.path.join(self.cache_dir, self.sigma_save_subfolder)        
        self.blend_method_dict = self.settings.get('blend_methods', {
            'karras': {'weight': 1.0, 'decay_pattern': 'zero', 'decay_mode': 'append', 'tail_steps': 1},
            'exponential': {'weight': 1.0, 'decay_pattern': 'zero', 'decay_mode': 'append', 'tail_steps': 1}
        })
        self.blend_methods = list(self.blend_method_dict.keys())
        self.blend_weights = [self.blend_method_dict[method]['weight'] for method in self.blend_methods]        
        self.loaded_sigmas = None
        self.sigma_sequences = {}        
        
        self.schedule_type = None
        self.suffix = None
        self.ext = None
        self._create_directories()
        self._finalize_init()
        
    def _create_directories(self):
        
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.sigma_save_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.extras_log_filename = os.path.join(
            self.settings.get('log_save_directory', 'modules/sd_simple_kes/image_generation_data'),
            f'all_extras_log_{timestamp}.txt'
        )

        
     
    def _finalize_init(self):
        
        self.prepass_save_file = self.build_sigma_cache_filename(
            steps=self.steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            schedule_type='karras',            
            decay_pattern=self.decay_pattern,
            cache_dir=self.sigma_save_folder,
            suffix='prepass',
            ext = 'pt'
        )
        self.final_save_file = self.build_sigma_cache_filename(
            steps=self.steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            schedule_type='karras',            
            decay_pattern=self.decay_pattern,
            cache_dir=self.sigma_save_folder,
            suffix='final',
            ext = 'pt'
        )
        self.load_blend_method_sigmas()
    def _load_config(self, config_path, **kwargs):
        self.logger = SharedLogger(debug=kwargs.get('debug', False))
        
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                return user_config or {}  # Always return a dict, even if empty
        except FileNotFoundError:
            self.logger.log(f"Config file not found: {config_path}. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            self.logger.log(f"Error loading config file: {e}")
            return {}

            
    def _validate_config_types(self):  
        '''
        Both corrects self.settings with an updated validated config, and also writes a corrected_user_config file with the correct types
        '''
        validated_settings = {}
        corrected_lines = []
        
        corrected_lines.append("# Corrected User Config (Invalid entries auto-corrected)\n")

        for key, (expected_type, default_value) in self._config_schema.items():
            value = self.settings.get(key, default_value)
            if isinstance(value, expected_type):
                validated_settings[key] = value
                corrected_lines.append(f"{key}: {value}")
            
            else:
                self.log(f"[Config Warning] Invalid type for '{key}': Expected {expected_type.__name__}, got {type(value).__name__}. Using default: {default_value}")
                validated_settings[key] = default_value
                corrected_lines.append(f"{key}: {default_value}  # Invalid type: {type(value).__name__}, replaced with default")

        # Save the corrected config with comments
        with open('corrected_user_config.yaml', 'w', encoding='utf-8') as f:
            f.write('\n'.join(corrected_lines))
        
        self.settings.update(validated_settings)
        
        for key, value in self.settings.items():
            setattr(self, key, value)

        
    def _log_extras_to_file(self, all_extras):
        """
        Logs the extras returned by each scheduler to a dedicated 'all_extras' log file.

        This method iterates through the list of extras for each blend method and writes them
        to a separate log file for easier tracking, debugging, and future analysis. 

        Parameters:
        ----------
        all_extras : list
            A list of extras returned by each scheduler, aligned with the blend_methods list.
            Each item in the list corresponds to the extras provided by a specific scheduler.

        Notes:
        -----
        - If extras are present, they are logged under their respective scheduler names.
        - If extras contain complex objects, the method attempts to serialize them using JSON.
        - Non-serializable extras are logged as raw text.

        Purpose:
        -------
        This log file is intended for developers to track additional outputs that are not 
        directly part of the sigma, tails, or decay sequences but may be useful for diagnostics, 
        metadata, or advanced scheduler behaviors.
        """
        with open(self.extras_log_filename, 'a', encoding='utf-8') as f:
            f.write("\n=== New Scheduler Extras ===\n")
            for method, extras in zip(self.blend_methods, all_extras):
                if extras:
                    try:
                        f.write(f"\nScheduler: {method}\n")
                        f.write(json.dumps(extras, indent=2))
                        f.write("\n")
                    except TypeError:
                        f.write(f"\nScheduler: {method}\n")
                        f.write(f"Extras (non-serializable): {extras}\n")
            f.write("\n============================\n")
    def __call__(self):
        # First pass: Run prepass to determine predicted_stop_step
        if not self.skip_prepass:
            self.prepass_compute_sigmas(
                steps=self.steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                device=self.device,
                skip_prepass=self.skip_prepass  
            )
            

        if self.load_prepass_sigmas:
            self.generate_sigmas_schedule(mode='prepass')

        if self.load_sigma_cache:
            self.generate_sigmas_schedule(mode='final')

        else:
            # Build sigma sequence directly (without prepass)
            self.config_values()
            self.generate_sigmas_schedule()

            if self.blending_mode == 'default':
                self.blend_sigma_sequence(                    
                    sigmas_karras= self.scheduler_registry.get('karras')(
                        steps=self.steps,
                        sigma_min=self.sigma_min,
                        sigma_max=self.sigma_max,
                        device=self.device,
                        decay_pattern=self.decay_pattern
                    )[2],
                    sigmas_exponential=self.scheduler_registry.get('exponential')(
                        steps=self.steps,
                        sigma_min=self.sigma_min,
                        sigma_max=self.sigma_max,
                        device=self.device,
                        decay_pattern=self.decay_pattern
                    )[2],
                    pre_pass=False,
                    blend_methods=self.blend_methods,
                    blend_weights=self.blend_weights
                )

            else:
                # For multi-method blending
                self.blend_sigma_sequence(                    
                    sigmas_karras=None,  # Not used in non-default mode
                    sigmas_exponential=None,  # Not used in non-default mode
                    pre_pass=False,
                    blend_methods=self.blend_methods,
                    blend_weights=self.blend_weights
                )

        sigmas = self.compute_sigmas(steps=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max, rho=self.rho, device=self.device)
        

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
        if self.debug:
            self.save_generation_settings()
        
        return sigmas

    def _safe_sigma_loader(self, cache_key):
        cache_folder = self.sigma_save_folder

        # Check if the folder exists and has files
        if not os.path.exists(cache_folder) or not os.listdir(cache_folder):
            self.log(f"[Cache Check] Cache folder {cache_folder} is empty or missing. Skipping load.")
            return None  # Signal to recompute

        # Check if matching file exists
        matching_files = [f for f in os.listdir(cache_folder) if cache_key in f and f.endswith('.pt')]

        if not matching_files:
            self.log(f"[Cache Check] No matching cache file found for key: {cache_key}. Skipping load.")
            return None  # Signal to recompute

        # If matching file found, load it
        filename = os.path.join(cache_folder, matching_files[0])
        self.log(f"[Cache Hit] Loading sigma cache from: {filename}")
        loaded_data = torch.load(filename, map_location=self.device)
        return loaded_data['sigma_values'].to(self.device)    
        
    def call_scheduler(self, method_name, *args, **kwargs):
        sigma_sequence = getattr(self, f"sigmas_{method_name}")  
        if sigma_sequence is None:
            self.log(f"No sigma sequence found for method: {method_name}")
            return None
        return sigma_sequence

    def is_sigma_randomized(self):
        return (
            self.settings.get('sigma_min_rand', False) or
            self.settings.get('sigma_max_rand', False) or
            self.settings.get('rho_rand', False) or 
            self.settings.get('sigma_max_enable_randomization_type', False) or 
            self.settings.get('sigma_min_enable_randomization_type', False) or 
            self.settings.get('rho_enable_randomization_type', False)  
        )
    

    def save_sigmas_as_csv(self, sigmas, filename):
        np.savetxt(filename, sigmas.cpu().numpy(), delimiter=",")
    
    def build_sigma_cache_filename(self, steps, sigma_min, sigma_max, rho=None, schedule_type='karras', decay_pattern='zero', cache_dir=r'modules\sd_simple_kes\cache', suffix=None, ext = None or 'txt'):
        if cache_dir is None:
            cache_dir = r'modules\sd_simple_kes\cache'
        if schedule_type == 'karras':
            base_filename = f'sigma_{schedule_type}_{steps}steps_rho{rho}_min{sigma_min}_max{sigma_max}_{decay_pattern}'
        else:
            base_filename = f'sigma_{schedule_type}_{steps}steps_min{sigma_min}_max{sigma_max}_{decay_pattern}'

        # If a suffix is provided, versioning applies
        if suffix:
            base_filename += f'_{suffix}'
            version = self.get_next_version_number(cache_dir, base_filename)
            if ext:
                version = self.get_next_version_number(cache_dir, base_filename, ext)
            filename = f'{version:03d}_{base_filename}.{ext}'
        else:
            # No versioning if suffix is not provided
            filename = f'{base_filename}.{ext}'
        
        return os.path.join(cache_dir, filename)
    
    def get_next_version_number(self, cache_dir, base_filename,ext=None):
        pattern = os.path.join(cache_dir, f'*_{base_filename}')
        if ext:
            pattern= os.path.join(cache_dir, f'*_{base_filename}.{ext}')
        existing_files = glob.glob(pattern)

        version_numbers = []
        for file in existing_files:
            match = re.search(r'(\d{3})_' + re.escape(base_filename), os.path.basename(file))
            if match:
                version_numbers.append(int(match.group(1)))

        if version_numbers:
            return max(version_numbers) + 1
        else:
            return 1
    
    def get_sigma_with_cache(self, steps, sigma_min, sigma_max, rho=7.0, device='cpu',
                         schedule_type='karras', decay_pattern=None, cache_dir=None, cache_file=None,
                         suffix=None, ext=None, mode=None, cache_key = None):
        self.steps = steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.device = device
        self.schedule_type = schedule_type
        self.decay_pattern = decay_pattern
        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.suffix = suffix
        self.ext = ext
        self.mode = mode
        self.cache_key = cache_key

        # Try to retrieve from in-memory cache first
        cached_sigmas = self.get_sigma_from_cache(cache_key)
        
        if cached_sigmas is not None:
            return cached_sigmas

        # If sigma is randomized → always generate new
        if self.is_sigma_randomized():
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        # If nothing is loaded yet → generate and cache
        if self.loaded_sigmas is None:
            _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
            self.loaded_sigmas = sigmas
            self.sigma_cache[cache_key] = sigmas
            return sigmas

        # Handle file cache modes
        if mode == 'prepass':
            self.cache_file = self.prepass_save_file
        elif mode == 'final':
            self.cache_file = self.final_save_file
        else:
            self.cache_file = self.build_sigma_cache_filename(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, cache_dir)

        # Load from file cache if enabled
        if mode in ['prepass', 'final'] and self.load_prepass_sigmas:
            loaded_sigmas = self.load_sigmas_with_hash_validation(
                filename=self.cache_file,
                steps=steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                rho=rho,
                device=device,
                schedule_type=schedule_type,
                decay_pattern=decay_pattern,
                cache_key = cache_key
            )

            if loaded_sigmas is not None:
                self.loaded_sigmas = loaded_sigmas
                self.sigma_cache[cache_key] = loaded_sigmas
                return loaded_sigmas.to(device)
            else:
                self.log("[Cache Recovery] Cache load failed. Recalculating sigma schedule.")
        '''
        # Cache miss → recalculate
        self.log(f"[Cache Miss] Recalculating sigma schedule for: {self.cache_file}")
        _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
        self.sigma_cache[cache_key] = sigmas
        '''
        return sigmas


    def load_sigmas_with_hash_validation(self, filename, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data=None, cache_key = None, suffix=None):
        if self.load_prepass_sigmas:
            if cache_key:
                try:
                    loaded_data = torch.load(filename, map_location=self.device)
                    self.loaded_sigmas = loaded_data['sigma_values'].to(self.device)
                    loaded_hash = loaded_data['sigma_hash']

                    expected_hash = self.generate_sigma_hash(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data, suffix)

                    if loaded_hash != expected_hash:
                        self.log(f"[Sigma Validator] Hash mismatch. Expected: {expected_hash}, Found: {loaded_hash}. Recalculating.")
                        return None  # Return None to signal the scheduler to recalculate
                    else:
                        self.log(f"[Sigma Validator] Hash validated successfully for file: {filename}")
                        return self.loaded_sigmas

                except Exception as e:
                    self.log("[Cache Recovery] Sigma cache invalid or missing. Recalculating sigmas.")            
                    _, _, _, sigmas = self._generate_sigmas(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern)
                    return sigmas

   
    def generate_sigma_hash(self, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, save_data=None, suffix=None):
        data_string = f'{steps}_{sigma_min}_{sigma_max}_{rho}_{device}_{schedule_type}_{decay_pattern}_{suffix}'
        hash_object = hashlib.sha256(data_string.encode())
        return hash_object.hexdigest()[:12]  # Use first 12 characters for compact ID
        
    def _generate_sigmas(self, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern=None, decay_mode=None, tail_steps=None):
        scheduler_func = self.scheduler_registry.get(schedule_type)

        if scheduler_func is None:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        tails, decay, extras, sigmas = self.call_scheduler_function(
            scheduler_func,
            steps=steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=device,
            decay_pattern=decay_pattern,
            decay_mode=decay_mode,
            tail_steps=tail_steps
        )

        return tails, decay, extras, sigmas

        

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
            for line in self.logger.prepass_log_buffer:
                f.write(f"{line}\n")
        self.log(f"[SimpleKEScheduler] Generation settings saved to {self.log_filename}")
            
        self.logger.log_buffer.clear() 
        self.logger.prepass_log_buffer.clear()
    
    def save_image_plot(self, sigs, i):
        graph_plot = plot_sigma_sequence(
            self.sigs[:i + 1],
            i,
            self.log_filename,
            self.graph_save_directory,
            self.graph_save_enable
        )
        self.log(f"Sigma sequence plot saved to {graph_plot}")

    
    
    
    def apply_global_randomization(self):
        """Force randomization for all eligible settings by enabling _rand flags and re-randomizing values."""
        # First pass: turn on all _rand flags if corresponding _rand_min/_rand_max exists
        for key in list(self.settings.keys()):
            if key.endswith("_rand_min") or key.endswith("_rand_max"):
                base_key = key.rsplit("_rand_", 1)[0]
                rand_flag_key = f"{base_key}_rand"
                self.settings[rand_flag_key] = True
        # Step 2: If global_randomize is active, re-randomize all eligible keys
        if self.global_randomize:           
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
    
    
    def resolve_blend_weights(self, blend_weights, blending_style):  
        if blending_style == 'softmax':
            # Softmax automatically normalizes weights per step
            blend_weights = torch.tensor(blend_weights)
            normalized_weights = torch.softmax(blend_weights, dim=0)
            return normalized_weights.tolist()

        elif blending_style == 'explicit':
            # Return raw weights, will manually normalize in blending step
            return blend_weights

        else:
            raise ValueError(f"Unknown blending_style: {blending_style}")
    
    def extract_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() > 1:
                return value.mean().item()  # or first element
            else:
                return value.item()
        return value  # Already a float

    def _call_legacy_mode(self, schedule_type):
        # Validate schedule_type
        if schedule_type not in ['karras', 'exponential']:
            self.log(f"[Legacy Mode] Unsupported schedule_type: {schedule_type}")
            return

        # Dynamically set target attribute
        target_attr = f"sigmas_{schedule_type}"

        scheduler_func = self.scheduler_registry.get(schedule_type)

        tails, decay, extras, sigmas = self.call_scheduler_function(
            scheduler_func,
            steps=self.steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=self.device,
            decay_pattern=self.decay_pattern
        )

        # Assign to self.sigmas_karras or self.sigmas_exponential dynamically
        setattr(self, target_attr, sigmas)

        self.log(f"[Legacy Mode] Loaded sigma sequence for {schedule_type}. Assigned to self.{target_attr}")


    def blend_sigma_sequence(self, sigmas_karras=None, sigmas_exponential=None, pre_pass=False, blend_methods=None, blend_weights=None):
        # Filter out schedulers with zero weight
        active_methods = [
            method for method, config in self.blend_method_dict.items() if config.get('weight', 1.0) > 0.0
        ]
        # Fallback if all weights are zero
        if not active_methods:
            self.log("[Blend Config] All weights are zero. Falling back to default blend (karras + exponential).")
            '''
            # Values set in init, placed here for reference
            self.blend_method_dict = {
                'karras': {'weight': 1.0, 'decay_pattern': 'zero', 'decay_mode': 'append', 'tail_steps': 1},
                'exponential': {'weight': 1.0, 'decay_pattern': 'zero', 'decay_mode': 'append', 'tail_steps': 1}
            }
            '''
            active_methods = list(self.blend_method_dict.keys())

        self.blend_methods = active_methods

        # Rebuild sigma lists        
        self.blend_weights = [self.blend_method_dict[m]['weight'] for m in self.blend_methods]

        # Edge Case: No active schedulers
        if len(self.blend_methods) == 0:
            raise ValueError("[SimpleKEScheduler] No active schedulers selected. Please check your blend configuration.")
            #Should never happen because we default to init value for default if 0.

        # Edge Case: Only one scheduler (direct usage)
        if len(self.blend_methods) == 1:
            self.log(f"[Blend] Only one active scheduler: {self.blend_methods[0]}. Skipping blending, using it directly.")
            self.sigs = self.sigma_sequences[self.blend_methods[0]]['sigmas']
            #return self.sigs  # Do not Exit early!! Continue function

        if len(self.blend_methods) == 2:
            self.blending_mode = 'smooth_blend'
        elif len(self.blend_methods) > 2:
            self.blending_mode = 'weights'

        
        if not self.allow_step_expansion and self.auto_mode_enabled:
            self.auto_mode_enabled = False
            self.log("[Auto Mode] Step expansion disallowed. Auto mode forcibly disabled.")

        self.progress = torch.linspace(0, 1, len(self.sigs)).to(self.device)
        self.blended_sigmas = []
        self.change_log = []
        self.relative_converged = False
        self.max_converged = False
        self.delta_converged = False
        self.early_stop_triggered = False
        
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
        if self.sigmas_exponential is None:
            self._call_legacy_mode(schedule_type='exponential')

        if self.sigmas_karras is None:
            self._call_legacy_mode(schedule_type='karras')
        
        self.prepass_blended_sigmas = []
        self.blended_sigma = None
        self.blended_sigmas=[]
        for i in range(len(self.sigs)):
            if self.step_progress_mode == "linear":
                progress_value = self.progress[i]
            elif self.step_progress_mode == "exponential":
                progress_value = self.progress[i] ** self.exp_power
            elif self.step_progress_mode == "logarithmic":
                progress_value = torch.log1p(self.progress[i] * (torch.exp(torch.tensor(1.0)) - 1))
            elif self.step_progress_mode == "sigmoid":
                progress_value = 1 / (1 + torch.exp(-12 * (self.progress[i] - 0.5)))
            else:
                progress_value = self.progress[i]  # Fallback to linear (previous version used)  
            
            self.dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]
            self.smooth_blend = torch.sigmoid((self.dynamic_blend_factor - self.blend_midpoint) * self.smooth_blend_factor)
            self.noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor   
            self.step_size = self.initial_step_size * (1 - progress_value) + self.final_step_size * progress_value * self.step_size_factor
            if self.blending_mode == 'default':
                # Classic default: Karras + Exponential only
                self.blended_sigma = self.sigmas_karras[i] * (1 - self.smooth_blend) + self.sigmas_exponential[i] * self.smooth_blend 
            
            if self.blending_mode == 'smooth_blend' or (self.blending_mode == 'auto' and len(self.blend_methods) == 2):
                # Smooth blend between exactly two methods
                sigma_seq_a = self.sigma_sequences[self.blend_methods[0]]['sigmas']
                sigma_seq_b = self.sigma_sequences[self.blend_methods[1]]['sigmas']


                self.blended_sigma = sigma_seq_a[i] * (1 - self.smooth_blend) + sigma_seq_b[i] * self.smooth_blend
                 
                           
            elif self.blending_mode == 'weights' or (self.blending_mode == 'auto' and len(self.blend_methods) > 2):
                
                
                if self.blend_weights is None:
                    self.blend_weights = [1.0] * len(self.all_sigmas)
                if self.blending_style is None:
                    self.blending_style = 'soft_max'

                # Resolve weights based on blending style
                resolved_blend_weights = self.resolve_blend_weights(self.blend_weights, self.blending_style)
                
                weighted_sum = sum(w * self.extract_scalar(s[i]) for w, s in zip(resolved_blend_weights, self.all_sigmas))


                total_weight = sum(resolved_blend_weights)
                self.blended_sigma = weighted_sum / total_weight
            
            for s in self.all_sigmas:
                self.log(f"[DEBUG]sigma sequence shape: {s.shape}")

            
            self.sigs[i] = self.blended_sigma * self.step_size * self.noise_scale
            self.change = torch.abs(self.sigs[i] - self.sigs[i - 1])
            # Safely extract scalar for both tensor and float
            self.change_log.append(self.extract_scalar(self.change))
            relative_sigma_progress = (self.blended_sigma - self.sigs[-1].item()) / self.blended_sigma                            
            recent_changes = torch.abs(torch.tensor(self.change_log[-5:]))
            max_change = torch.max(recent_changes).item()                        
            mean_change = torch.mean(recent_changes).item()
            #percent_of_threshold = (max_change / self.early_stopping_threshold) * 100                                        
            self.delta_change = abs(max_change - mean_change)
            #self.blended_sigmas.append(self.blended_sigma.item()) 
            self.blended_sigmas.append(self.extract_scalar(self.blended_sigma))

            # Check 1: Relative sigma progress
            self.relative_converged = relative_sigma_progress < 0.05
            # Check 2: Max recent sigma change
            self.max_converged = max_change < self.early_stopping_threshold
            # Check 3: Max-mean difference converged
            self.delta_converged = self.delta_change < self.recent_change_convergence_delta            
            
            if pre_pass:
                self.prepass_blended_sigmas=self.blended_sigmas.copy()
                self.prepass_blended_sigma = self.blended_sigma
                if i >= 2:
                    
                    sigma_rate = abs(self.prepass_blended_sigmas[i] - self.prepass_blended_sigmas[i - 1])
                    previous_sigma_rate = abs(self.prepass_blended_sigmas[i - 1] - self.prepass_blended_sigmas[i - 2])
                    if sigma_rate > previous_sigma_rate:
                        self.prepass_log(f"Sigma decline is slowing down → possible plateau at step {i+1}.")

                if i == 0:
                    self.prepass_log("\n--- Starting Pre-Pass Blending ---\n")  
                    step_label = "Prepass First Step"
                elif i == len(self.sigs) - 1:
                    step_label = "Prepass Last Step"
                else:
                    step_label = None

                if step_label:
                    self.prepass_log(f"[{step_label} - Step {i}/{len(self.sigs)}] Prepass Blended Sigma: {self.prepass_blended_sigma:.6f}, Final Sigma: {self.sigs[i]:.6f}")    
                    self.prepass_log(f"{step_label} Delta Converged: {self.delta_converged} delta_change: {self.delta_change:.6f}, Target Default Settings:{self.recent_change_convergence_delta}")
                    
                # Start checking for early stopping after minimum steps
                if i > self.safety_minimum_stop_step and len(self.change_log) > 10:                  
                    # Calculate variance and dynamic threshold
                    self.blended_tensor = torch.tensor(self.prepass_blended_sigmas) 
                    if self.device == 'cpu':
                        self.sigma_variance = np.var(self.prepass_blended_sigmas)
                    else: 
                        self.sigma_variance = torch.var(self.sigs).item()

                    self.min_sigma_threshold = self.sigma_variance * self.sigma_variance_scale  # scale factor can be tuned
                    self.prepass_log(f"\n--- Early Stopping Evaluation at Step {i} ---")
                    self.prepass_log(f"Current Blended Prepass Sigma: {self.prepass_blended_sigma:.6f}")
                    self.prepass_log(f"Sigma Variance: {self.sigma_variance:.6f}")
                    self.prepass_log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                    self.prepass_log(f"Max Recent Sigma Change: {max_change:.6f}")
                    self.prepass_log(f"Mean Recent Sigma Change: {mean_change:.6f}")
                    

                    # Reason for continuing (sigma still too high)
                    if self.prepass_blended_sigma > self.min_sigma_threshold:
                        self.prepass_log(f"Prepass Blended Sigma {self.prepass_blended_sigma:.6f} exceeds min sigma threshold {self.min_sigma_threshold:.6f} → Continuing.\n")
                                            
                    # Start Early Stopping Checks                    
                    if self.early_stopping_method == "mean":
                        mean_change = sum(self.change_log) / len(self.change_log)
                        if mean_change < self.early_stopping_threshold:
                            skipped_steps = len(self.sigs) - (i)
                            self.prepass_log(f"Early stopping triggered by mean at step {i}. Mean change: {mean_change:.6f}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")                                                

                    elif self.early_stopping_method == "max":
                        #max_change = max(self.change_log)
                        if max_change < self.early_stopping_threshold:
                            skipped_steps = len(self.sigs) - (i)
                            self.prepass_log(f"Early stopping triggered by mean at step {i}. Mean change: {max_change:.6f}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")                         
                                                    
                    elif self.early_stopping_method == "sum":
                        stable_steps = sum(
                            1 for j in range(1, len(self.change_log))
                            if abs(self.change_log[j]) < self.early_stopping_threshold * abs(self.sigs[j])
                        )
                        if stable_steps >= 0.8 * len(self.change_log):
                            skipped_steps = len(self.sigs) - (i)
                            self.prepass_log(f"Early stopping triggered by sum at step {i}. Stable steps: {stable_steps}/{len(self.change_log)}. Steps used: {i}/{len(self.sigs)}, steps skipped: {skipped_steps}")                                             
                            
                    if self.relative_converged and self.max_converged and self.delta_converged: 
                        self.early_stop_triggered = True                    
                        self.prepass_log(f"\n--- Early Stopping Evaluation at Step {i+1} ---")
                        self.prepass_log(f"Relative Sigma Progress: {relative_sigma_progress:.6f}")
                        self.prepass_log(f"Max Recent Sigma Change: {max_change:.6f}")
                        self.prepass_log(f"Mean Recent Sigma Change: {mean_change:.6f}")
                        self.prepass_log(f"Delta Change: {delta_change:.6f} (Target: {self.recent_change_convergence_delta})")                        
                        self.prepass_log(f"Early stopping criteria met at step {i+1} based on all convergence checks.")
                        self.predicted_stop_step = i
                        #self.steps = self.predicted_stop_step
                        self.save_image_plot(self.sigs, i)
                        break     
            
            
            # === Final Pass ===         
            if not pre_pass:               
                    
                if i == 0:
                    step_label = "First Step"
                    self.log("\n" + "=" * 10 + "\n[Start of Sigma Sequence Logging]\n" + "=" * 10)
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                         f"\nStep Size: {self.step_size:.6f}"
                         f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                         f"\nNoise Scale: {self.noise_scale:.6f}"
                         f"\nSmooth Blend: {self.smooth_blend:.6f}"
                         f"\nBlended Sigma: {self.blended_sigma:.6f}"
                         f"\nFinal Sigma: {self.sigs[i]:.6f}")
                elif i == len(self.sigs) // 2:
                    step_label = "Middle Step"
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                         f"\nStep Size: {self.step_size:.6f}"
                         f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                         f"\nNoise Scale: {self.noise_scale:.6f}"
                         f"\nSmooth Blend: {self.smooth_blend:.6f}"
                         f"\nBlended Sigma: {self.blended_sigma:.6f}"
                         f"\nFinal Sigma: {self.sigs[i]:.6f}")
                elif i == len(self.sigs) - 1:
                    step_label = "Last Step"
                    self.log(f"[{step_label} - Step {i}/{len(self.sigs)}]"
                         f"\nStep Size: {self.step_size:.6f}"
                         f"\nDynamic Blend Factor: {self.dynamic_blend_factor:.6f}"
                         f"\nNoise Scale: {self.noise_scale:.6f}"
                         f"\nSmooth Blend: {self.smooth_blend:.6f}"
                         f"\nBlended Sigma: {self.blended_sigma:.6f}"
                         f"\nFinal Sigma: {self.sigs[i]:.6f}")
                    self.log("\n" + "=" * 10 + "\n[End of Sigma Sequence Logging]\n" + "=" * 10)
                else:
                    step_label = None
                  
                if i > 0:
                    self.change = torch.abs(self.sigs[i] - self.sigs[i - 1])
                    #self.change_log.append(self.change.item())
                    self.change_log.append(self.extract_scalar(self.change))

                # Early Stopping Evaluation
                if i > self.safety_minimum_stop_step and len(self.change_log) > 5:                                 
                    final_target_sigma = self.sigs[-1].item()  # or use min(self.sigmas) if preferred                      
                    if self.blended_sigma != 0:
                        relative_sigma_progress = (self.blended_sigma - final_target_sigma) / self.blended_sigma
                    else:
                        relative_sigma_progress = 0  # Assume fully converged if blended_sigma is 0                                    
                    # Optional: Show variance but no need to stop on it
                    self.sigma_variance = torch.var(self.sigs).item() if self.device != 'cpu' else np.var(self.blended_sigmas)
                    self.log(f"Sigma Variance: {self.sigma_variance:.6f}")
            if self.graph_save_enable:
                self.save_image_plot(self.sigs, i)
                
        #apply tails and decay after the loop finishes
        # Finished core sigma blending  
        if not self.auto_mode_enabled:        
            if not pre_pass:  # Only extend in the final pass
                if self.apply_tail_steps:
                    for i, tail in enumerate(self.all_tails):
                        if tail is not None:
                            self.log(f"Appending tail from method: {self.blend_methods[i]}")
                            self.sigs = torch.cat([self.sigs, tail])

                if self.apply_decay_tail:
                    for i, decay in enumerate(self.all_decays):
                        if decay is not None:
                            self.log(f"Appending decay from method: {self.blend_methods[i]}")
                            self.sigs = torch.cat([self.sigs, decay])

                if self.apply_progressive_decay:
                    progressive_decay = None
                    total_weight = 0

                    for w, decay in zip(resolved_blend_weights, self.all_decays):
                        if decay is not None:
                            decay = decay[:len(self.sigs)]  # Ensure matching length
                            if progressive_decay is None:
                                progressive_decay = w * decay
                            else:
                                progressive_decay += w * decay
                            total_weight += w

                    if progressive_decay is not None and total_weight > 0:
                        progressive_decay /= total_weight
                        self.log("Applying progressive decay to sigma sequence.")
                        self.sigs = self.sigs * progressive_decay
                
                if self.apply_blended_tail:
                    blended_tail = None
                    total_weight = 0

                    for w, tail in zip(resolved_blend_weights, self.all_tails):
                        if tail is not None:
                            if blended_tail is None:
                                blended_tail = w * tail
                            else:
                                blended_tail += w * tail
                            total_weight += w

                    if blended_tail is not None and total_weight > 0:
                        blended_tail /= total_weight
                        self.log("Appending blended tail to sigma sequence.")
                        self.sigs = torch.cat([self.sigs, blended_tail])

        else:
            # Run Auto Mode stabilization sequence
            if len(self.sigs) > self.steps:
                self.auto_stabilization_sequence = []
                self.log(f"[Auto Mode] Sigma sequence length {len(self.sigs)} exceeds requested steps {self.steps}. Disabling auto stabilization.")
                self.auto_mode_enabled = False
                self.sigs = self.sigs[:self.steps]  # Force truncate to requested step count
                return self.sigs
            self.run_auto_stabilization(self.sigs)
        
        if pre_pass and self.early_stop_triggered:
            return self.sigs[:self.predicted_stop_step]  # Return only the usable sequence        
        else:
            return self.sigs
    def run_auto_stabilization(self):
        #This function works as intended, but is blocked by default if programs don't let schedulers create a sigma schedule longer than requested steps.
        if not self.allow_step_expansion:
            self.log("[Auto Mode] Step expansion is disabled by configuration. Skipping auto stabilization.")
            return self.sigs
        if self.allow_step_expansion:            
            unstable = self.detect_sequence_instability()

            if not unstable:
                self.log("[Auto Mode] Sigma sequence is already stable.")
                return

            self.log("[Auto Mode] Detected instability in sigma sequence. Starting stabilization sequence.")

            for method in self.auto_stabilization_sequence:
                if not unstable:
                    self.log(f"[Auto Mode] Sequence stabilized after {method}. Stopping further corrections.")
                    break

                if method == 'smooth_interpolation':
                    unstable = self.smooth_interpolation()

                elif method == 'append_tail':
                    unstable = self.append_tail()

                elif method == 'blend_tail':
                    unstable = self.blend_tail()

                elif method == 'apply_decay':
                    unstable = self.apply_decay()

                elif method == 'progressive_decay':
                    unstable = self.progressive_decay()

                else:
                    self.log(f"[Auto Mode] Unknown stabilization method: {method}")
    def detect_sequence_instability(self):
        delta_sigmas = self.sigs[:-1] - self.sigs[1:]
        second_deltas = torch.diff(delta_sigmas)

        steep_drop_detected = torch.any(delta_sigmas > self.auto_tail_threshold)
        jaggedness_score = torch.var(second_deltas[-5:]) if len(second_deltas) >= 5 else 0
        jagged_transition_detected = jaggedness_score > self.jaggedness_threshold

        if steep_drop_detected:
            self.log(f"[Auto Mode] Steep drop detected. Max drop: {torch.max(delta_sigmas).item():.6f}")
        if jagged_transition_detected:
            self.log(f"[Auto Mode] Jagged transition detected. Jaggedness score: {jaggedness_score:.6f}")

        return steep_drop_detected or jagged_transition_detected
    def smooth_interpolation(self):
        self.log("[Auto Mode] Applying smooth interpolation to last 5 steps.")
        if len(self.sigs) >= 5:
            start = self.sigs[-6].item()
            end = self.sigs[-1].item()
            interpolated = torch.linspace(start, end, steps=6, device=self.device)[1:]
            self.sigs[-5:] = interpolated

        return self.detect_sequence_instability()

    def append_tail(self):
        self.log("[Auto Mode] Attempting to append available tail.")
        if hasattr(self, 'all_tails') and self.all_tails:
            for tail in self.all_tails:
                if tail is not None:
                    tail = tail.to(self.device)
                    # If tail is longer than remaining sequence, trim
                    if tail.shape[0] > self.sigs.shape[0]:
                        tail = tail[:len(self.sigs)]
                    self.sigs = torch.cat([self.sigs, tail])
                    self.log("[Auto Mode] Appended tail to sigma sequence.")
                    break

        return self.detect_sequence_instability()


    def blend_tail(self):
        if not hasattr(self, 'all_tails') or not self.all_tails:
            self.log("[Auto Mode] No available tails to blend.")
            return self.detect_sequence_instability()

        self.log("[Auto Mode] Attempting to blend multiple tails.")
        blended_tail = None
        total_weight = 0

        for w, tail in zip(self.blend_weights, self.all_tails):
            if tail is not None:
                tail = tail.to(self.device)

                # Align length if needed
                if tail.shape[0] > self.sigs.shape[0]:
                    tail = tail[:len(self.sigs)]

                if blended_tail is None:
                    blended_tail = w * tail
                else:
                    blended_tail += w * tail
                total_weight += w

        if blended_tail is not None and total_weight > 0:
            blended_tail /= total_weight
            self.sigs = torch.cat([self.sigs, blended_tail])
            self.log("[Auto Mode] Appended blended tail to sigma sequence.")

        return self.detect_sequence_instability()


    def apply_decay(self):
        self.log("[Auto Mode] Attempting to append decay tails.")
        if hasattr(self, 'all_decays') and self.all_decays:
            for decay in self.all_decays:
                if decay is not None:
                    decay = decay.to(self.device)

                    # Align length if needed
                    if decay.shape[0] > self.sigs.shape[0]:
                        decay = decay[:len(self.sigs)]

                    self.sigs = torch.cat([self.sigs, decay])
                    self.log("[Auto Mode] Appended decay tail to sigma sequence.")
                    break

        return self.detect_sequence_instability()


    def progressive_decay(self):
        self.log("[Auto Mode] Applying progressive decay to sigma sequence.")
        progressive_decay = None
        total_weight = 0

        for w, decay in zip(self.blend_weights, self.all_decays):
            if decay is not None:
                decay = decay.to(self.device)

                # If the decay is too short, interpolate to match self.sigs length
                if decay.shape[0] != self.sigs.shape[0]:
                    decay = decay.view(1, 1, -1)  # Shape for interpolation
                    decay = F.interpolate(decay, size=self.sigs.shape[0], mode='linear', align_corners=False)
                    decay = decay.view(-1)

                if progressive_decay is None:
                    progressive_decay = w * decay
                else:
                    progressive_decay += w * decay

                total_weight += w

        if progressive_decay is not None and total_weight > 0:
            progressive_decay /= total_weight
            self.sigs = self.sigs * progressive_decay
            self.log("[Auto Mode] Applied progressive decay to sigma sequence.")

        return self.detect_sequence_instability()


    def load_blend_method_sigmas(self, mode=None):
        """Loads all sigma sequences for the blend_methods list based on current settings and mode."""
        self.all_sigmas = []
        

        for method in self.blend_methods:
            self.method_config = self.blend_method_dict[method]           
            self.method_config[method] = {
                'decay_pattern': self.method_config.get('decay_pattern', 'zero'),
                'decay_mode': self.method_config.get('decay_mode', 'blend'),
                'tail_steps': self.method_config.get('tail_steps', 1)
            }
            self.current_config = self.method_config[method]


            sigma_func = self.scheduler_registry[method]
            tails, decay, extras, sigmas = self.call_scheduler_function(
                self.scheduler_registry.get(method),
                steps=self.steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,  # Only passed if the scheduler accepts it
                device=self.device,
                decay_pattern=self.current_config['decay_pattern'],  # Method-specific
                decay_mode=self.current_config['decay_mode'],        # Method-specific
                tail_steps=self.current_config['tail_steps']         # Method-specific
            )
            self.sigma_sequences[method] = {
                    'sigmas': sigmas,
                    'tails': tails,
                    'decay': decay,
                    'extras': extras
                }
            setattr(self, f"sigmas_{method}", sigmas)                      
           
        self.all_sigmas = [self.sigma_sequences[method]['sigmas'] for method in self.blend_methods]
        self.all_tails = [self.sigma_sequences[method]['tails'] for method in self.blend_methods]
        self.all_decays = [self.sigma_sequences[method]['decay'] for method in self.blend_methods]
        self.all_extras = [self.sigma_sequences[method].get('extras', []) for method in self.blend_methods]
        self._log_extras_to_file(self.all_extras)
        self.all_sigmas.append(sigmas)

        # Optionally log which schedules were loaded
        self.log(f"Loaded sigma schedules for blend methods: {self.blend_methods} using mode: {mode}")
    def validate_and_align_sigmas(self):
        """
        Ensures all sigma sequences in self.all_sigmas are valid and have the same length.
        Pads shorter sequences with their last sigma.
        """
        if not self.all_sigmas or len(self.all_sigmas) == 0:
            raise ValueError("No sigma sequences were loaded for blending.")

        target_length = max(len(s) for s in self.all_sigmas)

        for idx, sigmas in enumerate(self.all_sigmas):
            if sigmas is None or len(sigmas) == 0:
                raise ValueError(f"Sigma sequence at index {idx} is invalid or empty: {sigmas}")

            if len(sigmas) < target_length:
                padding = torch.full((target_length - len(sigmas),), sigmas[-1]).to(sigmas.device)
                self.all_sigmas[idx] = torch.cat([sigmas, padding])

        self.log(f"Validated and aligned all sigma sequences to length {target_length}.")
     
    def generate_sigmas_schedule(self, mode=None):
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
        #self.cache_key= self.generate_sigma_hash(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device, self.schedule_type, self.decay_pattern, self.suffix or None)
        # ✅ Clean Mode Selection
        if mode == 'prepass':
            if self.load_prepass_sigmas:
                self.cache_file = self.prepass_save_file
            self.mode = 'prepass'

        elif mode == 'final':
            if self.load_sigma_cache:
                self.cache_file = self.final_save_file
            self.mode = 'final'

        else:
            self.mode = None
            self.cache_file = None  # Optional, for safety
        
        '''
        if self.cache_file:          
            sigmas = self.get_sigma_with_cache(
                steps=self.steps,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                rho=self.rho,
                device=self.device,                    
                decay_pattern=self.decay_pattern,
                cache_file=self.cache_file,
                mode=self.mode
                #cache_key = self.cache_key
            )
            return sigmas
        '''
            
        #else:
            #logic for multiple schedulers  
            #self.load_blend_method_sigmas(mode=self.mode)             
        self.load_blend_method_sigmas(mode=self.mode) 
        self.blend_pairs = []
        self.active_methods = [method for method in self.blend_methods if self.blend_method_dict[method].get('weight', 1.0) > 0]
        
        if self.blending_mode == 'default':
            self._call_legacy_mode(schedule_type='exponential')
            self._call_legacy_mode(schedule_type='karras')            
            
            self.blend_pairs = []
            self.blend_pairs.append({
                'method_label': 'method_a',
                'method': 'karras',
                'sigmas': self.sigmas_karras
            })
            self.blend_pairs.append({
                'method_label': 'method_b',
                'method': 'exponential',
                'sigmas': self.sigmas_exponential
            })
            
           # ✅ Optional: Pad if needed (if you know some might be misaligned)
            max_length = max(len(pair['sigmas']) for pair in self.blend_pairs)

            for pair in self.blend_pairs:
                if len(pair['sigmas']) < max_length:
                    padding = torch.full((max_length - len(pair['sigmas']),), pair['sigmas'][-1]).to(pair['sigmas'].device)
                    pair['sigmas'] = torch.cat([pair['sigmas'], padding])

            self.log(f"All sigma sequences aligned to length: {max_length}")

            # ✅ For legacy compatibility
            sigmas_a = self.blend_pairs[0]['sigmas']
            sigmas_b = self.blend_pairs[1]['sigmas']
            label_a = self.blend_pairs[0]['method']
            label_b = self.blend_pairs[1]['method']
            if sigmas_a is None:
                raise ValueError(f"Sigmas {label_a} failed to generate or assign correctly.")
            if sigmas_b is None:
                raise ValueError(f"Sigmas {label_b} failed to generate or assign correctly.")
        else:
            if len(self.active_methods) == 1:
                # Only one method, assign both to the same method
                self.blend_pairs = []
                method = self.active_methods[0]
                self.blend_pairs.append({
                    'method_label': 'method_a',
                    'method': method,
                    'sigmas': self.sigma_sequences[method]['sigmas']
                })

            elif len(self.active_methods) >= 2:
                # Build blend_pairs dynamically for all active methods
                self.blend_pairs = []
                for idx, method in enumerate(self.active_methods):
                    self.blend_pairs.append({
                        'method_label': f'method_{chr(97 + idx)}',  # method_a, method_b, etc.
                        'method': method,
                        'sigmas': self.sigma_sequences[method]['sigmas']
                    })

                # Validation checks (loop version)
                for pair in self.blend_pairs:
                    if pair['sigmas'] is None:
                        raise ValueError(f"Sigmas {pair['method']} failed to generate or assign correctly.")


        # Skip length matching if only 1 method is enabled        
        if len(self.blend_pairs) > 1:
            target_length = min(len(pair['sigmas']) for pair in self.blend_pairs)

            # Trim all to target length
            for pair in self.blend_pairs:
                pair['sigmas'] = pair['sigmas'][:target_length]

            # Find the max length (in case any sequence is longer after trimming)
            max_length = max(len(pair['sigmas']) for pair in self.blend_pairs)

            for pair in self.blend_pairs:
                if len(pair['sigmas']) < max_length:
                    padding = torch.full((max_length - len(pair['sigmas']),), pair['sigmas'][-1]).to(pair['sigmas'].device)
                    pair['sigmas'] = torch.cat([pair['sigmas'], padding])

            self.log(f"All sigma sequences aligned to length: {max_length}")
            
            self.sigs = torch.zeros(target_length, device=self.blend_pairs[0]['sigmas'].device)
        else:
            # Only one method, set self.sigs directly
            self.sigs = self.blend_pairs[0]['sigmas'].clone()
             
       
        '''
        # Now it's safe to compute sigs        
        start = math.log(self.sigma_max)
        end = math.log(self.sigma_min)
        #self.sigs = torch.linspace(start, end, self.steps, device=self.device).exp()       
        if self.sigs is None or self.force_rebuild_sigs:
            self.sigs = torch.linspace(start, end, self.steps, device=self.device).exp()


        
        # Ensure sigs contain valid values before using them
        if torch.any(self.sigs > 0):  
            self.sigma_min, self.sigma_max = self.sigs[self.sigs > 0].min(), self.sigs.max()            
        else:
            # If sigs are all invalid, set a safe fallback
            self.sigma_min, self.sigma_max = self.min_threshold, self.min_threshold              
            self.log(f"Debugging Warning: No positive sigma values found! Setting fallback sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
              
            return {
                'karras': self.sigmas_karras,
                'exponential': self.sigmas_exponential,
                'blend_methods': self.blend_methods,
                'all_sigmas': self.all_sigmas,
                'sigs': self.sigs
            }
       
        #sigma_lengths = [len(self.sigma_sequences[method]['sigmas']) for method in self.blend_methods]
        #if len(set(sigma_lengths)) > 1:  # There are mismatched lengths
            #self.validate_and_align_sigmas()
            #self.sigs = torch.zeros(self.steps, device=self.device)
        sigma_lengths = [len(self.sigma_sequences[method]['sigmas']) for method in self.blend_methods]
        if len(set(sigma_lengths)) > 1:
            self.log("[Sigma Alignment] Detected mismatched sigma sequence lengths. Aligning...")
            self.validate_and_align_sigmas()
            
        return {
            'blend_methods': self.blend_methods,
            'all_sigmas': self.all_sigmas,
            'sigs': self.sigs
        }
        '''
        # Now it's safe to compute sigs        
        #start = math.log(self.sigma_max)
        #end = math.log(self.sigma_min)
        #self.sigs = torch.linspace(start, end, self.steps, device=self.device).exp()
        
        
        '''
        if torch.any(self.sigs > 0):
            self.sigma_min, self.sigma_max = self.sigs[self.sigs > 0].min(), self.sigs.max()
        else:
            # If sigs are all invalid, set a safe fallback
            self.sigma_min = self.min_threshold
            self.sigma_max = self.min_threshold
            self.log(f"Debugging Warning: No positive sigma values found! Setting fallback sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")

            return {
                'blend_methods': self.blend_methods,
                'all_sigmas': self.all_sigmas,
                'sigs': self.sigs
            }

        return {
            'blend_methods': self.blend_methods,
            'all_sigmas': self.all_sigmas,
            'sigs': self.sigs
        }
        '''
       
        if not torch.any(self.sigs > 0):
            self.sigma_min = self.min_threshold
            self.sigma_max = self.min_threshold
            self.log(f"Debugging Warning: No positive sigma values found! Setting fallback sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
        else:
            self.sigma_min = self.sigs[self.sigs > 0].min()
            self.sigma_max = self.sigs.max()

        return {
            'blend_methods': self.blend_methods,
            'all_sigmas': self.all_sigmas,
            'sigs': self.sigs
        }



    def call_scheduler_function(self, scheduler_func, **kwargs):
        """
        Safely calls a scheduler function with dynamic argument filtering and flexible return handling.

        This method ensures that only the parameters accepted by the scheduler function are passed.
        It automatically handles scheduler functions that may return:
            - Only the sigma sequence
            - A tuple with (tails, sigmas)
            - A tuple with (tails, decay, sigmas)
            - A tuple with additional items (extras) before sigmas

        The method always assumes the last returned item is the sigma sequence, with optional
        items preceding it.

        Parameters:
        ----------
        scheduler_func : callable
            The scheduler function to be invoked. It may accept various arguments such as steps, sigma_min,
            sigma_max, rho, device, decay_pattern, etc.
        **kwargs : dict
            Arbitrary keyword arguments. Only those accepted by the scheduler function will be passed.

        Returns:
        -------
        tuple
            A 4-tuple containing:
                - tails : Any (optional, can be None)
                    The tail component of the sigma schedule, if provided.
                - decay : Any (optional, can be None)
                    The decay component of the sigma schedule, if provided.
                - extras : list
                    Any additional return values provided by the scheduler function, beyond tails and decay.
                - sigmas : Any
                    The sigma sequence, always assumed to be the last item returned by the scheduler function.

        Raises:
        ------
        ValueError
            If the scheduler function returns an empty tuple.

        Notes:
        -----
        This method allows future schedulers to return additional optional data without breaking the calling pattern.
        """
        valid_params = inspect.signature(scheduler_func).parameters
        filtered_args = {k: v for k, v in kwargs.items() if k in valid_params}

        result = scheduler_func(**filtered_args)

        if isinstance(result, dict):
            tails = result.get('tails', None)
            decay = result.get('decay', None)
            sigmas = result.get('sigmas')
            extras = result.get('extras', [])

            if sigmas is None:
                raise ValueError("Scheduler function must return a 'sigmas' key.")

            return tails, decay, extras, sigmas

        # Legacy support: fallback to tuple unpacking if needed
        if not isinstance(result, tuple):
            return None, None, [], result

        if len(result) == 0:
            raise ValueError(f"Scheduler function returned an empty tuple. This is not allowed.")

        sigmas = result[-1]
        optional_returns = result[:-1]

        tails = optional_returns[0] if len(optional_returns) > 0 else None
        decay = optional_returns[1] if len(optional_returns) > 1 else None
        extras = optional_returns[2:] if len(optional_returns) > 2 else []

        return tails, decay, extras, sigmas

    def config_values(self):    
        #Ensures sigma_min is always less than sigma_max for edge cases       
        if self.sigma_min >= self.sigma_max:
            correction_factor = random.uniform(0.01, 0.99)
            old_sigma_min = self.sigma_min
            self.sigma_min = self.sigma_max * correction_factor
            self.log(f"[Correction] sigma_min ({old_sigma_min}) was >= sigma_max ({self.sigma_max}). Adjusted sigma_min to {self.sigma_min} using correction factor {correction_factor}.")

        self.log(f"Final sigmas: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}")
        
        
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
        
        
        valid_methods = ['mean', 'max', 'sum']
        if self.early_stopping_method not in valid_methods:
            self.log(f"[Config Correction] Invalid early_stopping_method: {self.early_stopping_method}. Defaulting to 'mean'.")
            self.early_stopping_method = 'mean'
       
    def prepass_compute_sigmas(self, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, suffix=None, cache_key = None, skip_prepass = False)->torch.Tensor:   
       
        '''
        if self.load_prepass_sigmas:
            if cache_key:
                self._safe_sigma_loader(cache_key)
                #self.load_or_regenerate_sigmas(cache_key)
                loaded_data = torch.load(self.cache_file.replace('.txt', '.pt'), map_location=self.device)
                
                sigmas = loaded_data['sigma_values'].to(self.device)
                self.loaded_sigmas = sigmas  # No need to call torch.tensor again

                loaded_hash = loaded_data['sigma_hash']

                steps = loaded_data['steps']
                sigma_min = loaded_data['sigma_min']
                sigma_max = loaded_data['sigma_max']
                rho = loaded_data['rho']
                device = loaded_data['device']
                schedule_type = loaded_data['schedule_type']
                decay_pattern = loaded_data['decay_pattern']

                restored_config = loaded_data['full_config']  


                # Optionally overwrite current settings with restored settings
                self.settings.update(restored_config)            
                
               #self.load_sigmas_with_hash_validation(self, loaded_data, steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, cache_key, suffix=None)
                self.log(f"[Cache Loaded] Sigma schedule, hash, and config loaded from: {self.cache_file.replace('.pt', '.txt')}")
        '''
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
        
        if self.steps is None:
            raise ValueError("Number of steps must be provided.")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.config_values()        
        self.generate_sigmas_schedule(mode='prepass')
         
        self.predicted_stop_step = self.steps if None else self.original_steps
        if self.sharpen_last_n_steps > len(self.sigs):
            self.sharpen_last_n_steps = len(self.sigs)
            self.log(f"[Sharpening Notice] Requested last {self.sharpen_last_n_steps} steps exceeds sequence length. Using entire sequence instead.")
        
        self.visual_sigma = max(0.8, self.sigma_min * self.min_visual_sigma)
        
        self.blend_sigma_sequence(             
            sigmas_karras=None,
            sigmas_exponential=None,
            pre_pass = True,
            blend_methods=self.blend_methods,
            blend_weights = self.blend_weights
        )        
        if torch.isnan(self.sigs).any() or torch.isinf(self.sigs).any():
            raise ValueError("Invalid sigma values detected (NaN or Inf).")
        final_steps = self.sigs[:self.predicted_stop_step].to(self.device) 
        # Store the results for later use in compute_sigmas
        self.final_steps = final_steps
        if self.blending_mode == 'default':
            self.final_sigmas_karras = self.sigmas_karras
            self.final_sigmas_exponential = self.sigmas_exponential        
            self.log(f" Final Steps = {self.final_steps}. Predicted_stop_step = {self.predicted_stop_step}. Original requested steps = {self.steps}")
            self.log(f"final sigmas karras: {self.final_sigmas_karras}")  
        else:
            # For multi-method blending
            self.final_sigmas_blended = torch.tensor(self.blended_sigmas, device=self.device)

            self.log(f" Final Steps = {self.final_steps}. Predicted_stop_step = {self.predicted_stop_step}. Original requested steps = {self.steps}")
            self.log(f"final blended sigmas: {self.final_sigmas_blended}")
            
            # Optionally log the contributing sigma sequences for debugging
            for idx, (method, sigmas) in enumerate(zip(self.blend_methods, self.all_sigmas)):
                self.log(f"Method: {method}, Sigma sequence: {sigmas}")
        
        
        '''
        # Build cache key 
        sigma_hash = self.generate_sigma_hash(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, suffix=None)

        if self.save_prepass_sigmas:
            save_data = {
                'sigma_values': sigmas.cpu(),  # Keep as tensor
                'sigma_hash': sigma_hash,
                'steps': steps,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'rho': rho,
                'device': device,
                'schedule_type': schedule_type,
                'decay_pattern': decay_pattern,
                'full_config': self.settings  # Save as raw dict
            }
            
            # Save directly with torch.save in .pt format
            torch.save(save_data, self.cache_file)  # Assuming self.cache_file has .pt extension
            self.log(f"[Sigma Saver] Final sigmas saved to: {self.cache_file}")
        '''
    def load_or_regenerate_sigmas(self, cache_key):
        if self.load_sigma_cache and cache_key:
            try:
                loaded_data = torch.load(self.cache_file, map_location=self.device)
                sigmas = loaded_data['sigma_values'].to(self.device)

            except FileNotFoundError:
                self.log(f"[Cache Warning] Cache file not found: {self.cache_file}")
                self.log(f"[Cache Recovery] Automatically recomputing sigma schedule.")
                _, _, _, sigmas = self._generate_sigmas(
                    self.steps,
                    self.sigma_min,
                    self.sigma_max,
                    self.rho,
                    self.device,
                    self.schedule_type,
                    self.decay_pattern
                )

        # Recompute if cache is disabled or failed
        _, _, _, sigmas = self._generate_sigmas(
            self.steps,
            self.sigma_min,
            self.sigma_max,
            self.rho,
            self.device,
            self.schedule_type,
            self.decay_pattern
        )
        '''
        if self.save_prepass_sigmas:
            # Optional: Cache the recomputed sigma schedule
            torch.save(save_data, self.prepass_save_file)
            self.log(f"[Sigma Saver] Final sigmas saved to: {self.prepass_save_file}")
        
        #self.sigma_cache[cache_key] = sigmas
        '''
        return sigmas

    def compute_sigmas(self, steps, sigma_min, sigma_max, rho, device, schedule_type=None, decay_pattern=None, cache_key=None)->torch.Tensor:  
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
        '''
        if self.load_sigma_cache and cache_key:
            sigmas = self._safe_sigma_loader(cache_key)
            if sigmas is None:
                self.log(f"[Cache Recovery] No valid cache found for key: {cache_key}. Recomputing sigma schedule.")
                _, _, _, sigmas = self._generate_sigmas(
                    self.steps,
                    self.sigma_min,
                    self.sigma_max,
                    self.rho,
                    self.device,
                    self.schedule_type,
                    self.decay_pattern
                )
            else:
                self.log(f"[Cache Hit] Sigma schedule successfully loaded from cache.")
                self.sigs = sigmas
        '''        
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
        self.config_values() 
        self.generate_sigmas_schedule(mode='final') 
        if hasattr(self, 'final_sigmas_karras'):
            self.sigs = torch.zeros_like(self.final_sigmas_karras).to(self.device)        
        else:
            self.sigs = torch.zeros_like(self.sigmas_karras).to(self.device)
                      
        self.blend_sigma_sequence(            
            sigmas_karras=self.final_sigmas_karras if hasattr(self, 'final_sigmas_karras') else self.sigmas_karras,
            sigmas_exponential=self.final_sigmas_exponential if hasattr(self, 'final_sigmas_exponential') else self.sigmas_exponential,
            pre_pass=False,
            blend_methods=self.blend_methods,
            blend_weights = self.blend_weights
            
        )
        self.sigma_variance = torch.var(self.sigs).item()          
        if self.sharpen_mode in ['last_n', 'both']:
            if self.sigma_variance < self.sigma_variance_threshold:
                # Apply full sharpening
                self.sharpen_mask = torch.where(self.sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
                sharpen_indices = torch.where(self.sharpen_mask < 1.0)[0].tolist()
                self.sigs = self.sigs * self.sharpen_mask
                self.log(f"[Sharpen Mask] Full sharpening applied (low variance). Steps: {sharpen_indices}")
            else:
                # Apply sharpening only to the last N steps
                recent_sigs = self.sigs[-self.sharpen_last_n_steps:]
                sharpen_mask = torch.where(recent_sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
                sharpen_indices = torch.where(sharpen_mask < 1.0)[0].tolist()
                self.sigs[-self.sharpen_last_n_steps:] = recent_sigs * sharpen_mask

                # Now loop per step if desired (safely inside this block)
                for j in range(len(self.sigs) - self.sharpen_last_n_steps, len(self.sigs)):
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
        
        '''
        sigma_hash = self.generate_sigma_hash(steps, sigma_min, sigma_max, rho, device, schedule_type, decay_pattern, suffix=None)
        if self.settings.get('save_sigma_cache', False):
            save_data = {
                'sigma_values': self.sigs.cpu().tolist(),
                'sigma_hash': sigma_hash,
                'steps': steps,
                'sigma_min': sigma_min,
                'sigma_max': sigma_max,
                'rho': rho,
                'device': device,
                'schedule_type': schedule_type,
                'decay_pattern': decay_pattern,
                'full_config': json.dumps(self.settings)
            }
        
            
            if self.save_sigma_cache:            
                torch.save(save_data, self.final_save_file)
                self.log(f"[Sigma Saver] Final sigmas saved to: {self.final_save_file}")
        '''
        #self.log(f"[DEBUG]Final Output: Skip Prepass: {self.skip_prepass}. Original requested steps: {self.original_steps}. Self.steps = {self.steps} for tensor sigs: {self.sigs})")
        return self.sigs.to(self.device)
        
    def get_sigma_from_cache(self, cache_key):
        """
        Safely retrieves a sigma sequence from cache.
        Always returns a detached copy to prevent in-place modification of cached data.
        """
        if cache_key in self.sigma_cache:
            cached_sigmas = self.sigma_cache[cache_key]
            self.log(f"[Cache Hit] Returning cached sigma sequence for key: {cache_key}")

            # If tensor, clone and detach
            if isinstance(cached_sigmas, torch.Tensor):
                return cached_sigmas.clone().detach().to(self.device)

            # If list, deep copy
            elif isinstance(cached_sigmas, list):
                import copy
                return copy.deepcopy(cached_sigmas)

            # If other types, return as-is (optional: tighten later)
            else:
                return cached_sigmas

        else:
            self.log(f"[Cache Miss] Cache key not found: {cache_key}")
            return None
