import torch
import logging
from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential
import os
import yaml
import random
from datetime import datetime
import warnings
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# simple_karras_exponential_scheduler.py
# Developed by KittensX#
# License: MIT License


def get_random_or_default(scheduler_config, key_prefix, default_value, global_randomize):
    """Helper function to either randomize a value based on conditions or return the default."""
    
    # Determine if we should randomize based on global and individual flags
    randomize_flag = global_randomize or scheduler_config.get(f'{key_prefix}_rand', False)

    if randomize_flag:
        # Use specified min/max values for randomization if they exist, else use default range
        rand_min = scheduler_config.get(f'{key_prefix}_rand_min', default_value * 0.8)
        rand_max = scheduler_config.get(f'{key_prefix}_rand_max', default_value * 1.2)
        value = random.uniform(rand_min, rand_max)
        logging.info(f"Randomized {key_prefix}: {value}")
    else:
        # Use default value if no randomization is applied
        value = default_value
        logging.info(f"Using default {key_prefix}: {value}")

    return value
	
class ConfigManagerYaml:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config_data = self.load_config()  # Initialize config_data here

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                return user_config
        except FileNotFoundError:
            logging.info(f"Config file not found: {self.config_path}. Using empty config.")
            return {}
        except yaml.YAMLError as e:
            logging.error(f"Error loading config file: {e}")
            return {}
            
#ConfigWatcher monitors changes to the config file and reloads during program use (so you can continue work without resetting the program)        
class ConfigWatcher(FileSystemEventHandler):
    """Parameters are dynamically updated if the config file changes during execution."""
    def __init__(self, config_manager, config_path):
        self.config_manager = config_manager
        self.config_path = config_path

    def on_modified(self, event):
        if event.src_path == self.config_path:
            logging.info(f"Config file {self.config_path} modified. Reloading config.")
            self.config_manager.config_data = self.config_manager.load_config()  
            
def start_config_watcher(config_manager, config_path):
    event_handler = ConfigWatcher(config_manager, config_path)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(config_path), recursive=False)
    observer.start()
    return observer
    
# If user config is provided, update default config with user values
# Define the correct path for the config file
config_path = os.path.join(os.path.dirname(__file__), 'kes_config', 'simple_kes_scheduler.yaml')
config_manager = ConfigManagerYaml(config_path) 
  
# Start watching for config changes
observer = start_config_watcher(config_manager, config_path)

def simple_karras_exponential_scheduler(
    n, device, sigma_min=0.01, sigma_max=50, start_blend=0.1, end_blend=0.5, 
    sharpness=0.95, early_stopping_threshold=0.01, update_interval=10, initial_step_size=0.9, 
    final_step_size=0.2, initial_noise_scale=1.25, final_noise_scale=0.8, smooth_blend_factor=11, step_size_factor=0.8, noise_scale_factor=0.9, randomize=False, user_config=None
):    
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
    config_path = os.path.join(os.path.dirname(__file__), 'simple_kes_scheduler.yaml')
    config = config_manager.load_config()
    scheduler_config = config.get('scheduler', {})
    if not scheduler_config:
        warnings.warn("Scheduler configuration is missing from the config file. Using default values.")
    
    # Global randomization flag
    global_randomize = scheduler_config.get('randomize', False)
    
    #debug_log("Entered simple_karras_exponential_scheduler function")    
    default_config = {
        "debug": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",        
        "sigma_min": 0.01,
        "sigma_max": 50, #if sigma_max is too low the resulting picture may be undesirable. 
        "start_blend": 0.1,
        "end_blend": 0.5,
        "sharpness": 0.95,
        "early_stopping_threshold": 0.01,
        "update_interval": 10,
        "initial_step_size": 0.9,
        "final_step_size": 0.2,
        "initial_noise_scale": 1.25,
        "final_noise_scale": 0.8,
        "smooth_blend_factor": 11,
        "step_size_factor": 0.8, #suggested value to avoid oversmoothing
        "noise_scale_factor": 0.9, #suggested value to add more variation 
        "randomize": False,
        "sigma_min_rand": False,
        "sigma_min_rand_min": 0.001,
        "sigma_min_rand_max": 0.05,
        "sigma_max_rand": False,
        "sigma_max_rand_min": 0.05,
        "sigma_max_rand_max": 0.20,
        "start_blend_rand": False,
        "start_blend_rand_min": 0.05,
        "start_blend_rand_max": 0.2, 
        "end_blend_rand": False,
        "end_blend_rand_min": 0.4,
        "end_blend_rand_max": 0.6,
        "sharpness_rand": False,
        "sharpness_rand_min": 0.85,
        "sharpness_rand_max": 1.0,
        "early_stopping_rand": False,
        "early_stopping_rand_min": 0.001,
        "early_stopping_rand_max": 0.02,
        "update_interval_rand": False,
        "update_interval_rand_min": 5,
        "update_interval_rand_max": 10,
        "initial_step_rand": False,
        "initial_step_rand_min": 0.7,
        "initial_step_rand_max": 1.0,
        "final_step_rand": False,
        "final_step_rand_min": 0.1,
        "final_step_rand_max": 0.3,
        "initial_noise_rand": False,
        "initial_noise_rand_min": 1.0,
        "initial_noise_rand_max": 1.5,
        "final_noise_rand": False,
        "final_noise_rand_min": 0.6,
        "final_noise_rand_max": 1.0,
        "smooth_blend_factor_rand": False,
        "smooth_blend_factor_rand_min": 6,    
        "smooth_blend_factor_rand_max": 11,   
        "step_size_factor_rand": False,
        "step_size_factor_rand_min": 0.65,
        "step_size_factor_rand_max": 0.85,   
        "noise_scale_factor_rand": False,
        "noise_scale_factor_rand_min": 0.75,
        "noise_scale_factor_rand_max": 0.95,    
    }    
    logging.info(f"Default Config create {default_config}")
    config = config_manager.load_config().get('scheduler', {})
    if not config:
        warnings.warn("Scheduler configuration is missing from the config file. Using Default Values")
    
    # Log loaded YAML configuration
    logging.info(f"Configuration loaded from YAML: {config}")
    
    for key, value in config.items():
        if key in default_config:
            default_config[key] = value  # Override default with YAML value
            logging.info(f"Overriding default config: {key} = {value}")
        else:
            logging.info(f"Ignoring unknown config option: {key}")
        
    logging.info(f"Final configuration after merging with YAML: {default_config}")

    global_randomize = default_config.get('randomize', False)
    logging.info(f"Global randomization flag set to: {global_randomize}")

    logging.info(f"Config loaded from yaml {config}")
   
    # Now using default_config, updated with valid YAML values
    logging.info(f"Final Config after overriding: {default_config}")
    
    # Example: Reading the randomization flags from the config
    randomize = config.get('scheduler', {}).get('randomize', False)
    
    # Use the get_random_or_default function for each parameter
    #if randomize = false, then it checks for each variable for randomize, if true, then that particular option is randomized, with the others using default or config defined values.NOTE! if randomize is true, there is no need to make each variable true. If randomize is false, then only the values which are true will be randomized. If you don't want randomization, all flags should be false, and randomize should be false. If you want some to be randomized, randomize should be false, and variables should be true. 
    sigma_min = get_random_or_default(config, 'sigma_min', sigma_min, global_randomize)
    sigma_max = get_random_or_default(config, 'sigma_max', sigma_max, global_randomize)
    start_blend = get_random_or_default(config, 'start_blend', start_blend, global_randomize)
    end_blend = get_random_or_default(config, 'end_blend', end_blend, global_randomize)
    sharpness = get_random_or_default(config, 'sharpness', sharpness, global_randomize)
    early_stopping_threshold = get_random_or_default(config, 'early_stopping', early_stopping_threshold, global_randomize)
    update_interval = get_random_or_default(config, 'update_interval', update_interval, global_randomize)
    initial_step_size = get_random_or_default(config, 'initial_step', initial_step_size, global_randomize)
    final_step_size = get_random_or_default(config, 'final_step', final_step_size, global_randomize)
    initial_noise_scale = get_random_or_default(config, 'initial_noise', initial_noise_scale, global_randomize)
    final_noise_scale = get_random_or_default(config, 'final_noise', final_noise_scale, global_randomize)
    smooth_blend_factor = get_random_or_default(config, 'smooth_blend_factor', smooth_blend_factor, global_randomize)
    step_size_factor = get_random_or_default(config, 'step_size_factor', step_size_factor, global_randomize)
    noise_scale_factor = get_random_or_default(config, 'noise_scale_factor', noise_scale_factor, global_randomize)
    
       
    # Expand sigma_max slightly to account for smoother transitions
    sigma_max = sigma_max * 1.1
    logging.info(f"Using device: {device}")
    # Generate sigma sequences using Karras and Exponential methods
    sigmas_karras = get_sigmas_karras(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    sigmas_exponential = get_sigmas_exponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    config = config_manager.config_data.get('scheduler', {})            
    # Match lengths of sigma sequences
    target_length = min(len(sigmas_karras), len(sigmas_exponential))  
    sigmas_karras = sigmas_karras[:target_length]
    sigmas_exponential = sigmas_exponential[:target_length]
              
    logging.info(f"Generated sigma sequences. Karras: {sigmas_karras}, Exponential: {sigmas_exponential}")
    if sigmas_karras is None:
        raise ValueError("Sigmas Karras:{sigmas_karras} Failed to generate or assign sigmas correctly.")
    if sigmas_exponential is None:    
        raise ValueError("Sigmas Exponential: {sigmas_exponential} Failed to generate or assign sigmas correctly.")
          
    try:
        pass
    except Exception as e:
        error_log(f"Error generating sigmas: {e}")
   
  
    # Define progress and initialize blend factor
    progress = torch.linspace(0, 1, len(sigmas_karras)).to(device)
    logging.info(f"Progress created {progress}")
    logging.info(f"Progress Using device: {device}")
    
    sigs = torch.zeros_like(sigmas_karras).to(device)
    logging.info(f"Sigs created {sigs}")
    logging.info(f"Sigs Using device: {device}")

    for i in range(len(sigmas_karras)):
        # Adaptive step size and blend factor calculations
        step_size = initial_step_size * (1 - progress[i]) + final_step_size * progress[i] * step_size_factor  # 0.8 default value Adjusted to avoid over-smoothing
        logging.info(f"Step_size created {step_size}"   )
        dynamic_blend_factor = start_blend * (1 - progress[i]) + end_blend * progress[i]
        logging.info(f"Dynamic_blend_factor created {dynamic_blend_factor}"  )
        noise_scale = initial_noise_scale * (1 - progress[i]) + final_noise_scale * progress[i] * noise_scale_factor  # 0.9 default value Adjusted to keep more variation
        logging.info(f"noise_scale created {noise_scale}"   )
        
        # Calculate smooth blending between the two sigma sequences
        smooth_blend = torch.sigmoid((dynamic_blend_factor - 0.5) * smooth_blend_factor) # Increase scaling factor to smooth transitions more
        logging.info(f"smooth_blend created {smooth_blend}"   )
        
        # Compute blended sigma values
        blended_sigma = sigmas_karras[i] * (1 - smooth_blend) + sigmas_exponential[i] * smooth_blend
        logging.info(f"blended_sigma created {blended_sigma}"   )
        
        # Apply step size and noise scaling
        sigs[i] = blended_sigma * step_size * noise_scale

    # Optional: Adaptive sharpening based on sigma values
    sharpen_mask = torch.where(sigs < sigma_min * 1.5, sharpness, 1.0).to(device)
    logging.info(f"sharpen_mask created {sharpen_mask} with device {device}"   )
    sigs = sigs * sharpen_mask
    
    # Implement early stop criteria based on sigma convergence
    change = torch.abs(sigs[1:] - sigs[:-1])
    if torch.all(change < early_stopping_threshold):
        logging.info("Early stopping criteria met."   )
        return sigs[:len(change) + 1].to(device)
    
    if torch.isnan(sigs).any() or torch.isinf(sigs).any():
        raise ValueError("Invalid sigma values detected (NaN or Inf).")

    return sigs.to(device)
