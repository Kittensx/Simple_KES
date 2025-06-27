# SimpleKEScheduler

**Hybrid Sigma Scheduler for Stable Diffusion**  
_A blend of Karras & Exponential scheduling with adaptive, randomized control._


---

## 📌 What is it?

**SimpleKEScheduler** is a custom hybrid scheduler designed to replace or augment the default schedulers used in Stable Diffusion pipelines. It combines **Karras-style sigma sampling** with **exponential decay**, allowing for dynamic control over noise, sharpness, and step transitions.

This scheduler is highly configurable, supports structured randomization, and was built for developers and users who want fine-tuned control or experimental behavior beyond standard sampling techniques.

---
## Versions 1.2 Changelog
 - Version 1.2 Adds a new **prepass system** that allows the scheduler to automatically adjust the number of steps based on how quickly the image starts to stabilize. This means it might take an image with requested steps of 50 and instead use 35 because it estimates faster convergence based off of generated sigma schedules. In the second pass, we pass the steps from prepass into the primary scheduler as the steps and generate a new sigma sequence off of the new step count. 
 - Purpose of version 1.2: revamp how early_stopping_threshold works. In previous version it existed but didn't work. This version was created specifically to make it work. 
 
### New Features
- A new folder named ***simple_kes_v2*** has been dropped into both the A1111 and Forge folders for v1.2 The code inside is meant to replace the current version 1, and the folder names should be  renamed to drop the _v1.2 and placed into the proper "modules" folder in both programs
- Early stopping methods added: mean, max, and sum. In my tests, all 3 methods worked the same in that I did not notice a visual change, but we did see diifferent log outputs as to why it stopped. Examples: "Early stopping triggered by mean method. Mean change: 2.1957881450653076", "Early stopping triggered by max method. Max change: 11.830986022949219", "Early stopping triggered by sum method. Stable steps: 25 / 25"
- Early stopping will now occur when sigmas start to converge
- An option to turn early stopping off has also been added to allow you to run it in version 1 mode. It isn't the exact same as version 1, but it skips prepass function which uses the early stopping methods to reduce step count.
- New config options meant to affect early stopping have been added
- More detailed logging support has been added if enabled.
- A graph can be generated if enabled to show where the steps converge.
- I have tested and updated the default_config values for: sigma_scale_factor, sigma_auto_mode (use sigma_min), rho, sigma_min, sigma_max, start_blend, end_blend, sharpness, and early_stopping_threshold, and I have updated the recommended default values and rand_min/ rand_max values. As long as you stay within those defaults you should be ok. If you go below the rand_min, you will likely not see any visible changes. If you go above the rand_max you will probably notice negative things which may degrade image quality. For example, you can increase sharpness to 15 and you will start to notice very big changes - almost like a different seed. But if you go up to 50 you will have a warped or fuzzy image. Sharpness doesn't even get applied above 0.95, so increasing it above that goes beyond it's intended purpose. Sharpness below 0.75 was not noticeable during my tests.
- Step progress modes have been added: linear (default) is what we used in the previous version. You can also try: exponential, logarithmic, or sigmoid. Each has their own formula and affects how noise is scheduled and how fast it might be removed.
- Early stopping is ***experimental*** and performs well without considering hires.
#### Linear
A smooth, predictable pacing where each step changes by the same amount.
 ```python
progress_value = self.progress[i]
```

#### Exponential
Stronger texture retention early, fast cleanup later.
```python
progress_value = self.progress[i] ** self.settings.get("exp_power", 2)
```
Higher exp_power → even steeper curve.


#### Logarithmic
Faster image structure formation early, more careful refinement later.
```python
 progress_value = torch.log1p(self.progress[i] * (torch.exp(torch.tensor(1.0)) - 1))
```

#### Sigmoid
Natural feeling pacing, mimics many physical processes. Great for creating balanced transitions.
```python
progress_value = 1 / (1 + torch.exp(-12 * (self.progress[i] - 0.5)))
```

#### Summary
- The progress curve affects how aggressively or softly your step size changes over time.
- This influences how noise is removed and how details are formed.

##### Exponential → might create sharper textures and more punchy changes late in the sequence.

##### Logarithmic → might create smoother images with faster structure formation.

##### Sigmoid → might offer the most balanced pacing.


### Known Issues / Bugs
When using hires upscaling, the upscaler may not perform as well. Edges/details can appear washed out.
If early stopping occurs, tweak the settings for early_stopping, including sigma_variance_scale, safety_minimum_stop_step (try increasing), min_visual_sigma
- I have not tested the best values for initial_step_size, final_step_size, step_size_factor, initial_noise_scale, final_noise_scale, smooth_blend_factor, or noise_scale_factor, or the new early_stopping settings.





## Supported Stable Diffusion Projects (so far)
 - Automatic A1111 (for install click on 'A1111 - simple_kes' then follow the installation instructions)
 - Forge WebUi (for install click on Forge, then follow the installation instructions)
---

## 🚀 Features

- 🔀 **Hybrid Scheduling**: Smooth blend between Karras and exponential sigma schedules.
- ⚙️ **Parameter Blending**: `start_blend` and `end_blend` dynamically interpolate during sampling.
- 🎲 **Structured Randomization**: Enable full or partial parameter randomization for variability.
- 🪞 **Sharpening Controls**: Apply adaptive sharpening masks for low-sigma values.
- 🧠 **Early Stopping**: Optional convergence threshold for faster inference.
- 📦 **Modular Design**: Easy integration into pipelines, CLI tools, or training scripts.
- 🧪 **Inspired by A1111 & Diffusers**: Seamlessly replaces scheduler logic while remaining transparent to end-users.

---

## 📂 Installation

Clone this repo:

```bash
git clone https://github.com/Kittensx/Simple_KES
cd SimpleKEScheduler
```

Ensure your environment has the required packages:

```bash
pip install torch pyyaml
```

---

## 🧪 Usage

```python
from simple_kes import SimpleKEScheduler

# Load from YAML config file
scheduler = SimpleKEScheduler(
    steps=30,
    device='cuda',
    config='kes_config/default_config.yaml'  # or a custom YAML path
)

# Or use a Python dict:
config_dict = {
    "sigma_min": 0.01,
    "sigma_max": 1.0,
    "start_blend": 0.1,
    "end_blend": 0.5,
    "sharpness": 0.95,
    "early_stopping_threshold": 0.01,    
    "initial_step_size": 0.9,
    "final_step_size": 0.2,
    "initial_noise_scale": 1.25,
    "final_noise_scale": 0.8,
    "smooth_blend_factor": 8,
    "step_size_factor": 0.75,
    "noise_scale_factor": 0.85,
    "rho": 7.0,
    "global_randomize": False
}

scheduler = SimpleKEScheduler(steps=30, device='cuda', config=config_dict)

# Generate sigmas
sigmas = scheduler.compute_sigmas(steps=30, device='cuda')  # returns torch.FloatTensor
```

---

## 📁 File Structure

```
SimpleKEScheduler/
├── simple_kes.py               # Main scheduler logic
├── get_sigmas.py               # Karras & exponential helpers
├── __init__.py                 # Plugin hooks & config loader
├── kes_config/
│   └── default_config.yaml     # Default configuration file
└── proof it works.png          # Screenshot test proof
```

---

## 🔧 Configuration

Use the included `default_config.yaml` or create your own. All parameters support optional `_rand`, `_rand_min`, and `_rand_max` settings to enable structured randomness.

### Key Parameters

| Name                                        | Type  |            Description                      |
|---------------------------------------------|-------|---------------------------------------------|
| `sigma_min`   / `sigma_max`                 | float | Range of sigmas                             |
| `start_blend` / `end_blend`                 | float | Controls blending between methods           |
| `sharpness`                                 | float | Adaptive sharpening on low sigmas           |
| `initial_step_size` / `final_step_size`     | float | Controls progression pacing                 |
| `initial_noise_scale` / `final_noise_scale` | float | Noise scaling from start to end             |
| `global_randomize`                          | bool  | Force randomization on all supported params |

For the full schema, see `__init__.py:get_settings()`.

---

## 🧬 Why Use This?

- ✅ More flexible than static Karras or exponential schedulers
- ✅ Excellent for experiments, LoRA/finetuning, or style variation
- ✅ Compatible with CLI tools or training scripts
- ✅ Designed to “just work” with proper config input
- ✅ Structured randomization lets you simulate creative variability while preserving safety

---

## 💡 Example: Structured Randomization

Enable `global_randomize: true` in YAML or pass it in your config dict. You can also toggle flags like `sigma_min_rand` individually and specify ranges with `sigma_min_rand_min` / `sigma_min_rand_max`.

---


## 📜 License

MIT License 

---

## ✨ Credits

Built with inspiration from:

- [Karras et al. 2022](https://arxiv.org/abs/2206.00364)
- [HuggingFace diffusers](https://github.com/huggingface/diffusers)
- [AUTOMATIC1111 Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)

---

## 🙋‍♂️ Questions?

Feel free to open an issue or contribute improvements to the config system, plugin support, or integration with samplers!
