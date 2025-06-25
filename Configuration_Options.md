# Simple Karras Exponential Scheduler (SimpleKEScheduler)

A flexible sigma scheduler for Stable Diffusion pipelines supporting:

* Karras and Exponential scheduling
* Advanced blending and noise scaling
* Optional randomization (symmetric, asymmetric, or manual)
* Auto sigma min/max scaling

---

## Configuration Options

| Key                     | Description                                                            | Type   | Default        |
| ----------------------- | ---------------------------------------------------------------------- | ------ | -------------- |
| `sigma_min`             | Minimum sigma value (can be randomized or auto-controlled)             | float  | Required       |
| `sigma_max`             | Maximum sigma value (can be randomized or auto-controlled)             | float  | Required       |
| `rho`                   | Controls steepness of Karras curve                                     | float  | 7.0            |
| `sigma_auto_enabled`    | Enable auto mode to control sigma\_min or sigma\_max                   | bool   | False          |
| `sigma_auto_mode`       | `'sigma_min'` or `'sigma_max'` to control which sigma is auto-scaled   | string | `'sigma_min'`  |
| `sigma_scale_factor`    | Scale factor used when auto mode is enabled                            | float  | 1000           |
| `global_randomize`      | Enables randomization for all eligible parameters                      | bool   | False          |
| `randomization_type`    | `'asymmetric'`, `'symmetric'`, or `'off'` (can override per-parameter) | string | `'asymmetric'` |
| `randomization_percent` | Global randomization range percentage                                  | float  | 0.2            |
| `debug`                 | Enables detailed logging                                               | bool   | False          |

---

## Detailed Parameter Descriptions

### Sigma Parameters

* **`sigma_min` and `sigma_max`**

  * Define the minimum and maximum sigma values for noise scheduling.
  * Can be randomized or auto-controlled based on `sigma_auto_enabled` and `sigma_auto_mode`.
  * Auto mode recalculates one based on the other using `sigma_scale_factor`.

### Auto Sigma Control

* **`sigma_auto_enabled`**

  * If enabled, automatically controls either `sigma_min` or `sigma_max`.
* **`sigma_auto_mode`**

  * Determines which sigma is auto-calculated: `sigma_min` or `sigma_max`.
* **`sigma_scale_factor`**

  * Defines the scaling ratio when auto mode is active.

### Randomization Controls

* **`global_randomize`**

  * If enabled, activates randomization for all supported parameters (_rand_min and _rand_max values only)
* **`randomization_type`**

  *_randomization_type

### Specifies the type of advanced randomization.

##### Supported types:

symmetric → Random range is evenly distributed above and below the base value.

asymmetric → Random range is weighted with a larger upper range.

logarithmic → Random values are selected in log-space, useful for scaling-sensitive parameters.

exponential → Random values are generated using exponential growth based on a random base.

##### Short Name Aliases:

* s, sym → symmetric

* a, asym → asymmetric

* l, log → logarithmic

* e, exp → exponential

   **`randomization_percent`**

  * Defines the randomization range as a percentage of the base value.

### Blending Controls

* **`start_blend` and `end_blend`**

  * Control the dynamic blend factor between Karras and Exponential sigma schedules.
  * Blending is progress-based, transitioning smoothly from `start_blend` to `end_blend` over the steps.

### Sharpness Control

* **`sharpness`**

  * Applies an adaptive sharpening mask to the sigma sequence.
  * Increases detail retention when sigma values drop below a threshold.

### Step Size Controls

* **`initial_step_size` and `final_step_size`**

  * Define the step size at the beginning and end of the schedule.
  * Adjusted dynamically per step based on progress.

### Noise Scaling Controls

* **`initial_noise_scale` and `final_noise_scale`**

  * Define the noise scaling factor at the beginning and end of the schedule.
  * Adjusted dynamically per step based on progress.

### Smooth Blending Control

* **`smooth_blend_factor`**

  * Controls how smoothly the blend between Karras and Exponential schedules transitions.
  * Higher values create softer transitions.

### Adaptive Step and Noise Controls

* **`step_size_factor`**

  * Further scales the step size progression.
* **`noise_scale_factor`**

  * Further scales the noise scaling progression.

### Early Stopping Control

* **`early_stopping_threshold`**

  * Defines the minimum change required between steps.
  * If all step changes fall below this threshold, the scheduler triggers early stopping to optimize computation.

---

## Logging

* The scheduler writes generation logs to `modules/sd_simple_kes/image_generation_data`.
* Logs include full parameter tracking, sigma sequences, randomization details, and any auto mode adjustments.
