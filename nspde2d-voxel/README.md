# Neural SPDE, depth-stepped over 2D calorimeter showers

A Neural SPDE ([Salvi, Lemercier & Gerasimovics 2021](https://arxiv.org/abs/2110.10249)) applied to CaloChallenge Dataset 2 (SiW, fixed incidence).

**Idea**: treat the voxel grid `(Z, PHI, R) = (45, 16, 9)` as a (2+1)D space-time domain — `(PHI, R)` is space, depth `Z` is time. Generate a shower by solving, layer by layer:

```
dy = f(t, y) dt + g(t, y) · noise
```

- `y` = energy fraction per voxel (`shower_MeV / E_inc_MeV`), linear (not log-normalized like the sibling models)
- `f`, `g` ("drift"/"diffusion") = learned FNO-style spectral networks over `(PHI, R)`
- `noise` = pluggable driving process, chosen via `CFG["noise_type"]`

Extends `../sde_demo.ipynb`'s `CaloSDE` (fixed drift/diffusion formulas) by making everything learned. Single geometry (SiW) for now — multi-geometry conditioning like `cfm-dit_voxel` is a later step.

## Files

- `nspde2d_voxel.py` — all model code (import from here, don't redefine)
- `nspde2d-voxel.ipynb` — config, data loading, training/eval, plots
- `../utilities.py` — shared `CaloChallenge` data loader + plotting helpers

## Data

`y = shower_MeV / E_inc_MeV`, kept **linear** (not `[-1,1]` log like the other models) — the additive-drift/multiplicative-diffusion SPDE only makes physical sense in linear energy space. Each sample is the full 45-layer sequence (needed since training scores every transition at once).

## Model (`NeuralSPDE`)

- `SpectralConv2d` / `FourierLayer2d` — FNO spectral-conv layers over `(PHI, R)`, FiLM-conditioned on `(t, log10 E_inc)`
- `SpatialOperator` — shared backbone architecture for both `drift_net` and `diffusion_net`
- `InitialFieldNet` — predicts the layer-0 field distribution from `E_inc`

## Noise process — `CFG["noise_type"]`

Checkpoints are per-`noise_type` (`CFG["ckpt_dir"]`) — not interchangeable, since a checkpoint's `diffusion_net` output means something different for each:

| type | what it is | status |
|---|---|---|
| `"brownian"` | per-voxel Wiener process, exact training NLL | **broken** — variance collapses to 0, not fixed (see below) |
| `"colored_brownian"` | spatially-correlated Wiener (fixed power-law spectrum, `decay` not learned) | untested at scale |
| `"poisson"` | compound Poisson jumps — closest to real shot-noise physics, naturally sparse/non-negative | **the one in use** — trains stably |

## Training

`teacher_forced_loss` scores every real transition `t -> t+1` against Geant4 data in one batch (teacher forcing — no rollout at train time: every step's input is the real layer, never the model's own prior output, so one layer's error can't compound into the next). Generation (`NeuralSPDE.simulate`) does a real step-by-step Euler–Maruyama rollout instead — layer 44's input is whatever the model itself produced at layer 43. That gap between "trained on real inputs" vs. "generates from its own inputs" is exposure bias, and it's the recurring theme in "Known issues" below. Layer-0 is scored separately by `InitialFieldNet`'s own NLL.

**Training** (one big parallel pass, no loop over `t`):

```python
for epoch in range(epochs):
    for (y_real, E_inc) in train_loader:      # y_real: (batch, 45, H, W), real showers
        y_t, y_next = y_real[:, :-1], y_real[:, 1:]      # inputs / targets, all 44 transitions at once
        drift, rate = model(y_t, cond=(t, E_inc, cum_frac_from_y_real))
        predicted_next = y_t + drift + rate * quantum     # model's guess
        loss = huber(predicted_next, y_next) + aux_terms(...)   # scored against the REAL y_next
        loss.backward(); optimizer.step()
```

**Generation** (genuinely sequential, 44 steps):

```python
y = sample_initial(E_inc)
for t in range(44):
    drift, rate = model(y, cond=(t, E_inc, cum_frac_so_far))   # y is the MODEL's own last output
    y = y + drift + poisson_jump(rate)                          # becomes next step's input
    cum_frac_so_far += y.sum()
```

For `"poisson"`, the loss isn't a plain NLL — it's a Huber loss scaled by a ground-truth-derived scale, plus two auxiliary terms (`rate`/`quantum` regression target, and a per-shower total-energy match). Needed to stop the loss from collapsing early training by shrinking predicted variance to fit CaloChallenge's ~75% zero-inflated voxels — see git history / `teacher_forced_loss` docstring if you need the full debugging story.

### Loss function (`noise_type="poisson"`)

With $s=\sqrt{\max(y_t,y_{t+1})+\epsilon}$, $\mu = y_t + f_\theta(t,y_t,c)\,dt + \mathrm{softplus}(g_\theta(t,y_t,c))\,dt\,q$, and Huber $\mathrm{Hub}_\delta(z)=\tfrac12 z^2$ (else $\delta(|z|-\tfrac12\delta)$ for $|z|>\delta$):

$$
\mathcal{L} = \underbrace{\mathbb{E}\Big[\mathrm{Hub}_\delta\big(\tfrac{y_{t+1}-\mu}{s}\big)\Big]}_{\text{transition NLL}} + \underbrace{\mathbb{E}\Big[\mathrm{Hub}_\delta\big(\tfrac{\mathrm{softplus}(g_\theta)\,dt\,q-\mathrm{relu}(y_{t+1}-y_t)}{s}\big)\Big]}_{\text{rate/quantum target}} + \lambda_g\underbrace{\mathbb{E}_{\text{shower}}\Big[\mathrm{Hub}_\delta\big(\tfrac{\sum \mathrm{softplus}(g_\theta)\,dt\,q\;-\;\sum\mathrm{relu}(\Delta y)}{s_g}\big)\Big]}_{\text{per-shower energy match}}
$$
$$
+\; \lambda_d\,\mathbb{E}\Big[p\cdot\mathrm{relu}\big(\tfrac{f_\theta dt}{s}\big)^2 + \mathrm{relu}\big(\tfrac{-f_\theta dt}{s}\big)^2\Big] + \underbrace{\mathbb{E}\Big[\mathrm{Hub}_\delta\big(\tfrac{y_0-\mu_0}{s_0}\big)\Big]}_{\text{layer-0 mean}} + \underbrace{\mathbb{E}\Big[\mathrm{Hub}_\delta\big(\tfrac{\sigma_0-|y_0-\mu_0|}{s_0}\big)\Big]}_{\text{layer-0 std}}
$$

$c=(t,\log_{10}E_{\text{inc}},\text{cum\_frac})$; $q$ = learned jump quantum; $\mu_0,\sigma_0=$ `InitialFieldNet`$(E_{\text{inc}})$; $\lambda_g$=`global_reg_weight`, $\lambda_d$=`drift_reg_weight`, $p$=`drift_pos_mult`.

| term | motivation |
|---|---|
| transition NLL | scores the predicted step; ground-truth-derived $s$ (not the model's own std) blocks variance-collapse on zero-inflated voxels |
| rate/quantum target | $\mu$ is under-determined ($f_\theta$ alone could fit it) — gives the jump process its own gradient |
| per-shower energy match | per-voxel terms get diluted by the ~75% correctly-zero voxels; one number per shower fixes that |
| drift regularizer | keeps $f_\theta$ a small correction so it can't silently absorb mean-fitting and blow up over the 44-step rollout (exposure bias) |
| layer-0 mean/std | `InitialFieldNet`'s own NLL — nothing else scores layer 0 |

## Status

- `"poisson"`, 100 epochs, full 80k-shower dataset: trains stably, no collapse.
- Shower-max rise (onset, peak position) tracked reasonably well across `E_inc = 1`–`1000` GeV.
- **Containment shape is now learned** (cumulative-energy conditioning, see below) — z-profiles rise and decay past shower-max like real showers, confirmed across two retrains (2026-09-01, 2026-09-01 evening) at `E_inc = 1`–`1000` GeV.
- **Containment scale is still wrong, and three fixes tried so far haven't closed it**: cum_frac conditioning alone → ~10x under; adding scheduled sampling on top → ~30x over (and broke the shape again too); raising `global_reg_weight` 1.0 → 5.0 → no real change (~3-6% containment, same as the first attempt). All reverted/left as-is — see below. Paused here (2026-09-02) to think through the next step rather than starting more training runs blind.

## Known issues

- **Containment scale is wrong** — current open problem, now on its third attempt. Sequence of events:
  1. **Original bug**: neither `drift_net` nor the rate net had any signal for "how much energy have I already deposited," so nothing taught either one to stop — z-profiles plateaued/grew past shower-max instead of decaying.
     - Tried: penalizing positive `drift_dt` harder than negative (`drift_pos_mult`). Fixed the symptom but made results *worse* overall (per-hit EMD at 1 TeV: 135 → 670) — it just shifted the same problem onto the rate net, which then grew unboundedly instead (positive feedback: higher `y` → higher predicted rate → higher `y`). **Reverted** (`drift_pos_mult=1.0`, back to plain L2).
  2. **Fix: cumulative-energy conditioning.** `drift_net`/`diffusion_net` are now also conditioned on `cum_frac` (fraction of `E_inc` deposited so far, through and including the current layer) via a new `CumulativeEnergyEmbedding`, merged into `_cond_vec` alongside the existing `(t, log10 E_inc)` conditioning. In `teacher_forced_loss` this is computed exactly from `y_real`'s per-layer sums (teacher forcing); in `simulate` it's tracked as a running sum updated each rollout step. Existing checkpoints are **not** compatible (`cond_merge`'s input width changed). **Confirmed working for shape** on a fresh 100-epoch retrain (2026-09-01): z-profiles now decay past shower-max at every `E_inc` tested.
  3. **New bug this exposed**: that same retrain's total generated energy collapsed to ~2-9% of true across `E_inc = 1`–`1000` GeV. Root cause, by analogy with the original `drift_pos_mult` exposure-bias story: `cum_frac` is exact ground truth during teacher forcing, but at free-running generation it's built from the model's *own* rollout — so the moment early layers deposit even slightly too little, `cum_frac` lags behind where a real shower would be at that depth, an input the networks never saw during training. The observed failure (collapse toward under-deposition) is consistent with the networks reading a lagging `cum_frac` as "further along than I look" and suppressing further, which compounds every subsequent step — the same kind of runaway rollout feedback loop as the `drift_pos_mult=10` regression above, just running in the opposite direction.
  4. **Tried: scheduled sampling — made it worse, not better.** `teacher_forced_loss` gained a `scheduled_sampling_prob` argument (standard scheduled-sampling technique, Bengio et al. 2015, applied only to the `cum_frac` conditioning — `y_t` itself stays real/teacher-forced): for a random subset of the batch, `cum_frac` is instead computed from an actual no-grad self-rollout (`self.simulate`) with the model's current weights, ramped 0 → `scheduled_sampling_max_prob` over training via `nspde2d_voxel.scheduled_sampling_prob(cfg, epoch)`. Retrained 2026-09-01 evening (`start_epoch=30, ramp_epochs=40, max_prob=0.5`): containment went from ~10x under to **~30x over** across `E_inc=1`–`1000` GeV, and the shape regressed too (overshoots the real peak 5-15x, barely decays even by layer 44). From that run's own curves (wandb `78vkb5xr`): once `ss_prob` passed ~0.4 (epoch ~65-70), `grad_norm` spiked ~10x and `train_loss` got *worse* even as `test_loss` (pure teacher forcing, unaffected by scheduled sampling) kept improving to the best value of any run so far. Diagnosis: the self-rollout used to build `cum_frac` for the scheduled-sampling half of training was itself a snapshot of a model still in this architecture's known under-containment regime — so training on "real `y_t`, artificially-low self-rollout `cum_frac`" taught the networks to *compensate* for a low-looking `cum_frac` by injecting more energy, which combined with the rate net's known y-conditioned positive-feedback vulnerability (see `drift_pos_mult`'s entry above) into a runaway at generation time. **Reverted** (`scheduled_sampling_start_epoch=None` in `CFG`) — the mechanism is left implemented in `nspde2d_voxel.py` in case a more careful version (blending real/self-rollout `cum_frac` instead of full replacement, or gating on the self-rollout's own total-energy accuracy) is worth revisiting.
  5. **Tried: `global_reg_weight` 1.0 → 5.0 — no real change.** Retrained 2026-09-02 (`scheduled_sampling` still off): containment came back at ~2.5-5.6% across `E_inc=1`-`1000` GeV, essentially identical to the original cum_frac-only run, with the same correct rise/peak/decay shape. Confirms the caveat noted before running it: this term is teacher-forced-computed like everything else, so it can't see or correct the free-running rollout's compounding error — a 5x stronger weight on a proxy that isn't the actual failure mode doesn't move the actual failure mode. **Paused here** (2026-09-02) rather than guessing at a fourth training run — the next candidate is a direct rollout loss (backprop through an actual `self.simulate()` rollout, penalizing generated-total vs. `E_inc` directly), which is a real implementation effort (Poisson sampling isn't differentiable as-is, needs a reparameterization or moment-matched surrogate for the backward pass, plus gradient checkpointing for memory over a 44-step unroll) worth designing carefully before writing code, rather than rushing into a fourth blind attempt.
- **`"brownian"` noise is broken** — predicted std collapses toward 0 to cheaply fit zero-inflated data, which blows up the gradient on nonzero voxels. Deferred in favor of `"poisson"`, which doesn't have a freely-collapsible variance. Would need `std` floored/tied to a physical scale to fix.
- **`ColoredBrownianNoise`'s spatial correlation isn't learned** — `decay` is a hand-tuned float, not a trained parameter.
- **R axis (radial, bounded) is FFT'd as if periodic** — standard FNO practice for bounded domains, but a source of boundary error worth knowing about.
- **No explicit energy-conservation constraint** in the noise term itself (`sde_demo.ipynb` notes a candidate: zero the noise field's DC Fourier mode).
- `torch.compile` left off until the model/noise choice settles.

## Next steps

1. Design (then implement) a direct rollout loss — backprop through an actual `self.simulate()` rollout (or a truncated/checkpointed chunk of it) and penalize generated-total vs. `E_inc` directly, so training optimizes the real free-running objective instead of a teacher-forced proxy. Paused (2026-09-02) before starting this — worth thinking through the differentiability/memory design up front, since every proxy-conditioning fix tried so far (cum_frac, scheduled sampling, `global_reg_weight`) has only affected generation indirectly and twice made it worse.
2. Fix `"brownian"` variance collapse, or drop it.
3. Multi-geometry conditioning (`geom_id`, variable `phi`/`theta`), matching `cfm-dit_voxel`.