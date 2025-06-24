from typing import Optional, Dict, Any


RANDOMIZATION_TYPE_ALIASES = {
    'symmetric': 'symmetric', 'sym': 'symmetric', 's': 'symmetric',
    'asymmetric': 'asymmetric', 'assym': 'asymmetric', 'a': 'asymmetric',
    'off': 'off', 'none': 'off'
}

DEFAULT_RANDOMIZATION_TYPE = 'asymmetric'
DEFAULT_RANDOMIZATION_PERCENT = 0.2


def validate_config(config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:  
    updated_config = config.copy()

    def log(message):
        if logger:
            logger.log(message)
        else:
            print(message)

    # Correction for negative base values
    for key, value in config.items():
        if isinstance(value, (int, float)) and not key.endswith(('_rand', '_rand_min', '_rand_max', '_randomization_percent')):  # Skip randomization booleans and ranges
            if value < 0:
                updated_config[key] = abs(value)
                log(f"[Config Correction] {key} was negative. Converted to absolute value: {updated_config[key]}")

    # Existing randomization validation...
    for key, value in config.items():
        if key.endswith('_rand'):
            base_key = key.replace('_rand', '')

            if not isinstance(value, bool):
                updated_config[key] = False
                log(f"[Config Correction] {key} was not boolean. Set to False.")

            type_key = f"{base_key}_randomization_type"
            if type_key not in config:
                updated_config[type_key] = 'asymmetric'
                log(f"[Config Correction] {type_key} missing. Set to 'asymmetric'.")

            percent_key = f"{base_key}_randomization_percent"
            if percent_key not in config:
                updated_config[percent_key] = 0.2
                log(f"[Config Correction] {percent_key} missing. Set to 0.2.")

            min_key = f"{base_key}_rand_min"
            max_key = f"{base_key}_rand_max"
            if base_key in config:
                base_value = updated_config[base_key]  # Updated with absolute value if needed
                percent = updated_config[percent_key]

                if min_key not in config:
                    updated_config[min_key] = base_value * (1 - percent)
                    log(f"[Config Correction] {min_key} missing. Auto-calculated from base.")

                if max_key not in config:
                    updated_config[max_key] = base_value * (1 + percent)
                    log(f"[Config Correction] {max_key} missing. Auto-calculated from base.")

    log("[Config Validation] Config validated and missing values filled successfully.")
    return updated_config

