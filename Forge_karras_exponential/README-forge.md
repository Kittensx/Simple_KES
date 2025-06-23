# SimpleKEScheduler

**Hybrid Sigma Scheduler for Stable Diffusion**  
_A blend of Karras & Exponential scheduling with adaptive, randomized control._

---

## 📌 What is it?

**SimpleKEScheduler** is a custom hybrid scheduler designed to replace or augment the default schedulers used in Stable Diffusion pipelines. It combines **Karras-style sigma sampling** with **exponential decay**, allowing for dynamic control over noise, sharpness, and step transitions.

This scheduler is highly configurable, supports structured randomization, and was built for developers and users who want fine-tuned control or experimental behavior beyond standard sampling techniques.

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


## 📂 Installation – SimpleKEScheduler for A1111

To install and use the **SimpleKEScheduler** in your local A1111 (`stable-diffusion-webui`) environment:

### 1. 📁 Folder Setup

Copy the provided `sd_simple_kes` folder into the following location:

```
stable-diffusion-webui-forge/modules/
```

After doing this, you should now have:

```
stable-diffusion-webui-forge/modules/sd_simple_kes/
```

### 2. 🔧 Update `sd_schedulers.py`

Replace your existing `sd_schedulers.py` (located in `modules/`) with the included version from this package, or manually merge the required changes if you're using a modified version.

In particular, ensure:

- The following import is added:
  ```python
  from modules.sd_simple_kes.simple_kes import simple_kes_scheduler
  ```

- And the following scheduler is registered at the end of the `schedulers` list:
  ```python
  Scheduler('karras_exponential', 'Karras Exponential', simple_kes_scheduler)
  ```

If your A1111 version has diverged from the sample provided, you may need to adapt import paths or structure to fit your current layout. Just ensure `simple_kes_scheduler` is correctly imported and included in the `schedulers` list.

### 3. ✅ You're Done

Launch A1111 and select **"Karras Exponential"** as your sampler/scheduler.  
If everything is set up correctly, it will now use the SimpleKEScheduler for inference.

### 🛠 Need Help?

If you run into issues or are using a highly customized A1111 build, feel free to reach out. I'm happy to help troubleshoot integration or provide compatibility tips.


---

## 📁 File Structure

```
SimpleKEScheduler/
├── simple_kes.py               # Main scheduler logic
├── get_sigmas.py               # Karras & exponential helpers
├── kes_config/
└──    default_config.yaml     # Default configuration file
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