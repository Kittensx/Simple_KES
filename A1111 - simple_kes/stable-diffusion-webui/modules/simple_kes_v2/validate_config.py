from typing import Optional, Dict, Any

RANDOMIZATION_TYPE_ALIASES = {
    # Asymmetric
    'asymmetric': 'asymmetric', 'assym': 'asymmetric', 'a': 'asymmetric', 'asym': 'asymmetric', 'A': 'asymmetric',
    # Symmetric
    'symmetric': 'symmetric', 'sym': 'symmetric', 's': 'symmetric', 'S': 'symmetric',
    # Logarithmic
    'logarithmic': 'logarithmic', 'log': 'logarithmic', 'l': 'logarithmic', 'L': 'logarithmic',
    # Exponential
    'exponential': 'exponential', 'exp': 'exponential', 'e': 'exponential', 'E': 'exponential',
}

DEFAULT_RANDOMIZATION_TYPE = 'asymmetric'
DEFAULT_RANDOMIZATION_PERCENT = 0.2

# Base default values
BASE_DEFAULTS = {
    'sigma_min': 0.05,
    'sigma_max': 27.5,
    'start_blend': 0.1,
    'end_blend': 0.5,
    'sharpness': 1.0,
    'early_stopping_threshold': 0.01,
    'initial_step_size': 0.9,
    'final_step_size': 0.2,
    'initial_noise_scale': 1.25,
    'final_noise_scale': 0.8,
    'smooth_blend_factor': 9.0,
    'step_size_factor': 0.8,
    'noise_scale_factor': 0.8,
    'rho': 8.0
}

def validate_config(config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
    updated_config = config.copy()

    def log(message):
        if logger:
            logger.log(message)
        else:
            print(message)

    # Step 1: Set all base defaults if missing
    for key, base_value in BASE_DEFAULTS.items():
        if key not in updated_config:
            updated_config[key] = base_value
            log(f"[Config Correction] {key} missing. Set to base default: {base_value}")

        # Ensure _rand flag exists and is a boolean
        rand_flag = f"{key}_rand"
        if rand_flag not in updated_config or not isinstance(updated_config.get(rand_flag), bool):
            updated_config[rand_flag] = False
            log(f"[Config Correction] {rand_flag} missing or invalid. Set to False.")

        # Ensure _enable_randomization_type flag exists and is a boolean
        randomization_flag = f"{key}_enable_randomization_type"
        if randomization_flag not in updated_config or not isinstance(updated_config.get(randomization_flag), bool):
            updated_config[randomization_flag] = False
            log(f"[Config Correction] {randomization_flag} missing or invalid. Set to False.")

        # Ensure randomization_type exists
        randomization_type_key = f"{key}_randomization_type"
        if randomization_type_key not in updated_config:
            updated_config[randomization_type_key] = DEFAULT_RANDOMIZATION_TYPE
            log(f"[Config Correction] {randomization_type_key} missing. Set to '{DEFAULT_RANDOMIZATION_TYPE}'.")

        # Ensure randomization_percent exists
        randomization_percent_key = f"{key}_randomization_percent"
        if randomization_percent_key not in updated_config:
            updated_config[randomization_percent_key] = DEFAULT_RANDOMIZATION_PERCENT
            log(f"[Config Correction] {randomization_percent_key} missing. Set to {DEFAULT_RANDOMIZATION_PERCENT}.")

        # Ensure _rand_min and _rand_max exist
        min_key = f"{key}_rand_min"
        max_key = f"{key}_rand_max"
        percent = updated_config[randomization_percent_key]

        if min_key not in updated_config:
            updated_config[min_key] = updated_config[key] * (1 - percent)
            log(f"[Config Correction] {min_key} missing. Auto-calculated from base.")

        if max_key not in updated_config:
            updated_config[max_key] = updated_config[key] * (1 + percent)
            log(f"[Config Correction] {max_key} missing. Auto-calculated from base.")

    log("[Config Validation] Config validated and missing values filled successfully.")
    return updated_config
