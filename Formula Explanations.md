
# SimpleKEScheduler Formula Explanations

This document provides detailed explanations of key formulas used in the `SimpleKEScheduler` class.

---

## 1. Adaptive Step Size

```python
self.step_size = self.initial_step_size * (1 - self.progress[i]) + self.final_step_size * self.progress[i] * self.step_size_factor
```

**Explanation:**
- This line interpolates the step size from the initial to the final step over the diffusion process.
- Early steps favor the `initial_step_size`.
- Late steps favor the `final_step_size`, scaled by `step_size_factor`.

**Purpose:**
- Controls the intensity of each step dynamically.
- Prevents over-smoothing by adjusting step sizes as denoising progresses.

---

## 2. Dynamic Blend Factor

```python
self.dynamic_blend_factor = self.start_blend * (1 - self.progress[i]) + self.end_blend * self.progress[i]
```

**Explanation:**
- Smoothly transitions from `start_blend` to `end_blend` over time.

**Purpose:**
- Controls the weight shift between the Karras and Exponential schedules.

---

## 3. Adaptive Noise Scaling

```python
self.noise_scale = self.initial_noise_scale * (1 - self.progress[i]) + self.final_noise_scale * self.progress[i] * self.noise_scale_factor
```

**Explanation:**
- Scales noise contribution per step.
- Early steps have higher noise, later steps have reduced noise.

**Purpose:**
- Enables fine control over texture and randomness during generation.

---

## 4. Smooth Blending (Sigmoid)

```python
smooth_blend = torch.sigmoid((self.dynamic_blend_factor - 0.5) * self.smooth_blend_factor)
```

**Explanation:**
- The sigmoid ensures a smooth, curved transition instead of a sharp switch.
- `smooth_blend_factor` controls the steepness of the curve.

**Purpose:**
- Creates a gradual and controlled blend between Karras and Exponential schedules.

---

## 5. Blended Sigma Calculation

```python
blended_sigma = self.sigmas_karras[i] * (1 - smooth_blend) + self.sigmas_exponential[i] * smooth_blend
```

**Explanation:**
- Computes a weighted sigma combining both schedules.

**Purpose:**
- Seamlessly blends both sigma sequences to generate a hybrid denoising path.

---

## 6. Sigma Scaling

```python
sigs[i] = blended_sigma * self.step_size * self.noise_scale
```

**Explanation:**
- Scales the blended sigma by the adaptive step size and noise scale.

**Purpose:**
- Finalizes sigma magnitude for the step, factoring in both denoising speed and noise.

---

## 7. Adaptive Sharpening

```python
self.sharpen_mask = torch.where(sigs < self.sigma_min * 1.5, self.sharpness, 1.0).to(self.device)
sigs = sigs * self.sharpen_mask
```

**Explanation:**
- Applies a sharpening mask to sigmas lower than 1.5x the minimum sigma.

**Purpose:**
- Boosts sharpness selectively in low-sigma regions to preserve detail.

---

## 8. Early Stopping

```python
if torch.all(change < self.early_stopping_threshold):
    self.log("Early stopping criteria met.")
    return sigs[:len(change) + 1].to(self.device)
```

**Explanation:**
- Monitors sigma change between steps.
- If all changes are below the threshold, early stopping is triggered.

**Purpose:**
- Saves computation when the sigmas have effectively converged.
