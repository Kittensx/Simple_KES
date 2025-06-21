import os
import yaml
import torch

from simple_kes import SimpleKEScheduler

name = "simple_kes"
label = "Simple Karras Exponential Scheduler"
description = "Custom KES scheduler with exponential and Karras-style blending."

def _load_default_yaml_config():
    path = os.path.join(os.path.dirname(__file__), "kes_config", "default_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_default_settings():
    # Wrap it in a single named scheduler to avoid 'debug', 'rho', etc. issues
    return {"kes": _load_default_yaml_config()}
    
def get_label():
    return "Simple Karras Exponential Scheduler"

def get_description():
    return "A scheduler using Karras-style sigmas with blending options, inspired by A1111."

def get_settings(self) -> dict:
    """Returns the full configuration schema with types, defaults, and descriptions."""
    return {        
        "sigma_min": {
            "type": "float",
            "default": 0.01,
            "description": "Minimum sigma value. Lower values produce smoother images."
        },
        "sigma_max": {
            "type": "float",
            "default": 0.1,
            "description": "Maximum sigma value. Higher values allow more noise in early steps."
        },
        "device": {
            "type": "str",
            "default": "cuda",
            "description": "Device to run on: 'cuda' or 'cpu'."
        },
        "start_blend": {
            "type": "float",
            "default": 0.1,
            "description": "Initial blend factor between Karras and Exponential schedules."
        },
        "end_blend": {
            "type": "float",
            "default": 0.5,
            "description": "Final blend factor between Karras and Exponential schedules."
        },
        "sharpness": {
            "type": "float",
            "default": 0.95,
            "description": "Sharpening factor for low-sigma stages."
        },
        "early_stopping_threshold": {
            "type": "float",
            "default": 0.01,
            "description": "Threshold to trigger early stopping if sigmas converge."
        },
        "update_interval": {
            "type": "int",
            "default": 10,
            "description": "Steps between updates of the blend factor."
        },
        "initial_step_size": {
            "type": "float",
            "default": 0.9,
            "description": "Step size at the start of denoising."
        },
        "final_step_size": {
            "type": "float",
            "default": 0.2,
            "description": "Step size near the end of denoising."
        },
        "initial_noise_scale": {
            "type": "float",
            "default": 1.25,
            "description": "Amount of noise applied in early steps."
        },
        "final_noise_scale": {
            "type": "float",
            "default": 0.8,
            "description": "Amount of noise applied in later steps."
        },
        "randomize": {
            "type": "bool",
            "default": False,
            "description": "If true, enables global parameter randomization."
        },
        "bounds": {
          "type": "dict",
          "description": "Parameter-specific minimum and maximum bounds",
          "default":{
            "sigma_min_rand_min": 0.001,
            "sigma_min_rand_max": 0.05,
            "sigma_max_rand_min": 10,
            "sigma_max_rand_max": 60,
            "start_blend_rand_min": 0.05,
            "start_blend_rand_max": 0.2,
            "end_blend_rand_min": 0.4,
            "end_blend_rand_max": 0.6,
            "sharpness_rand_min": 0.85,
            "sharpness_rand_max": 1.0,
            "early_stopping_rand_min": 0.001,
            "early_stopping_rand_max": 0.02,
            "update_interval_rand_min": 5,
            "update_interval_rand_max": 10,
            "initial_step_rand_min": 0.7,
            "initial_step_rand_max": 1.0,
            "final_step_rand_min": 0.1,
            "final_step_rand_max": 0.3,
             "initial_noise_rand_min": 1.0,
            "initial_noise_rand_max": 1.5,
            "final_noise_rand_min": 0.6,
            "final_noise_rand_max": 1.0,
            "smooth_blend_factor_rand_min": 6,
            "smooth_blend_factor_rand_max": 11,
            "step_size_factor_rand_min": 0.65,
            "step_size_factor_rand_max": 0.85,
            "noise_scale_factor_rand_min": 0.75,
            "noise_scale_factor_rand_max": 0.95
          }
        },
        "randomization": {
            "type": "dict",
            "description": "Parameter-specific randomization flags and bounds.",
            "default": {
                "sigma_min_rand": False,                
                "sigma_max_rand": False,                
                "start_blend_rand": False,                
                "end_blend_rand": False,                
                "sharpness_rand": False,                
                "early_stopping_rand": False,                
                "update_interval_rand": False,                
                "initial_step_rand": False,                
                "final_step_rand": False,                
                "initial_noise_rand": False,               
                "final_noise_rand": False,                
                "smooth_blend_factor_rand": False,                
                "step_size_factor_rand": False,                
                "noise_scale_factor_rand": False                
            }
        }
    }

def scheduler(name="kes"):
    config = _load_default_yaml_config()
    ParamAliases().allow_custom_keys([
        "sigma_min", "sigma_max", "start_blend", "end_blend", "debug", "device",
        "noise_scale_factor_rand", "noise_scale_factor_rand_min",
        "noise_scale_factor_rand_max", "step_size_factor_rand_max",
        "early_stopping_threshold", "initial_step_size", "final_step_size", "global_randomize", "randomize"
        ])

    def scheduler_fn(steps: int):       
        # Pass as config dict, not a file path string
        scheduler_instance = SimpleKEScheduler(
            steps=steps,
            device=torch.device("cpu"),
            config=config 
        )

         # Call compute_sigmas and return result
        return scheduler_instance.compute_sigmas(steps, torch.device("cpu")).cpu().tolist()

    return scheduler_fn  # ✅ return the function itself, not a computed result


