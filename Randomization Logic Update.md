# Update: Enhanced Randomization Logic with `_enable_randomization_type`

## Overview

This update introduces a **more robust and explicit control** for parameter randomization using the `_enable_randomization_type` flag. This addition ensures **clear, deliberate activation** of advanced randomization types such as `symmetric`, `asymmetric`, `logarithmic`, and `exponential`.

Previously, randomization could unintentionally apply if `_rand` flags were enabled and randomization types existed in the configuration. This could cause unintended blending of randomization methods without an explicit on/off control.

---

## Key Changes

* ✅ Added `*_enable_randomization_type` flag for each parameter to **explicitly control** whether randomization types are applied.
* ✅ Updated `get_random_or_default` and related randomization methods to **enforce the requirement** that the randomization type is only active if `*_enable_randomization_type` is `True`.
* ✅ Improved config validation to auto-correct missing randomization keys and ensure base values and randomization percentages are present.
* ✅ Ensured safer default fallbacks when randomization flags or randomization types are not properly defined.

---

## How It Works Now

Each randomizable parameter now requires:

* `*_rand`: Enables min-max randomization.
* `*_enable_randomization_type`: Enables type-based randomization (symmetric, asymmetric, logarithmic, exponential).
* `*_randomization_type`: Specifies the randomization style.
* `*_randomization_percent`: Controls the degree of randomization.

**Example:**

```yaml
rho_rand: true
rho_rand_min: 6.0
rho_rand_max: 10.0
rho_enable_randomization_type: true
rho_randomization_type: asymmetric
rho_randomization_percent: 0.2
```

In this example:

* Min-max randomization is applied first.
* Then, an asymmetric randomization layer is applied **only because** `rho_enable_randomization_type` is set to `true`.

---

## Why This Change?

* 🔒 **Explicit Control:** Prevents accidental randomization when users may only want min-max behavior.
* 🛠️ **Safer Configs:** Forces a deliberate activation of complex randomization methods.
* 📜 **Cleaner Logs:** Makes it easier to track which randomization steps were actually applied.
* ⚙️ **Config Flexibility:** Allows for fine-tuning of which parameters should have advanced randomization versus simple min-max randomization.

---

## Compatibility

These changes are **backward compatible** if you add the new `_enable_randomization_type` keys to your existing configs. If missing, the system safely defaults to treating advanced randomization as disabled.

---

## Recommendation

Review your existing configs and ensure:

* You add `*_enable_randomization_type: true` if you want randomization types to be active.
* You test with the new system to confirm expected behavior.
