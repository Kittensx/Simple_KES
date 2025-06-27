# SimpleKEScheduler

**Hybrid Sigma Scheduler for Stable Diffusion**  
_A blend of Karras & Exponential scheduling with adaptive, randomized control._


---

## 📌 What is it?

**SimpleKEScheduler** is a custom hybrid scheduler designed to replace or augment the default schedulers used in Stable Diffusion pipelines. It combines **Karras-style sigma sampling** with **exponential decay**, allowing for dynamic control over noise, sharpness, and step transitions.

This scheduler is highly configurable, supports structured randomization, and was built for developers and users who want fine-tuned control or experimental behavior beyond standard sampling techniques.

---
## Versions 1.2 Changelog
 - Version 1.2 Adds a new **prepass system** that allows the scheduler to automatically adjust the number of steps based on how quickly the image starts to stabilize.
 - A New folder dropped into both the A1111 and Forge folders for v1.2 The code inside is meant to replace the current version 1, and the folder names should be  renamed to drop the _v1.2 and placed into the proper "modules" folder in both programs
 - New Features:
Early stopping methods added: mean, max, and sum
Early stopping will now occur when sigmas start to converge
An option to turn early stopping off has also been added to allow you to run it in version 1 mode. It isn't the exact same as version 1, but it skips prepass function which uses the early stopping methods to reduce step count.
New config options meant to affect early stopping have been added
More detailed logging support has been added if enabled.
A graph can be generated if enabled to show where the steps converge.
I have tested and updated the default_config values for: 

 - Known Issues / Bugs
When using hires upscaling, the upscaler may not perform as well. Edges/details can appear washed out.
If early stopping occurs, tweak the settings for early_stopping, including sigma_variance_scale, safety_minimum_stop_step (try increasing), min_visual_sigma



__
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
