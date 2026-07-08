# Conditional Flow Matching + DiT, jointly trained on two real geometries

Conditional Flow Matching (Lipman et al., 2023, https://arxiv.org/abs/2210.02747; rectified-flow
path per Liu et al., 2022, https://arxiv.org/abs/2209.03003) applied to 3D voxelized calorimeter
shower data, with the diffusion U-Net replaced by a CaloArt-inspired pure Diffusion Transformer
(DiT) backbone, **jointly trained on two real, independently-sourced geometries**:

- **`geom_id=0`** — CaloChallenge Dataset 2 (SiW), fixed incidence (`θ=π/2, φ=0`)
- **`geom_id=1`** — LEMURS FCCeeALLEGRO, variable incidence angle `(φ, θ)` per sample

This started as a clone of `../ddim-t_voxel/` (see that README for the original single-geometry,
energy-only version), then was extended for real geometry conditioning once it turned out
FCCeeALLEGRO data was actually fully present locally (the third candidate, LEMURS ODD, is not —
its file is 0 bytes, never downloaded — so this uses the two geometries that are genuinely there).

## Why these two geometries, and why not more

- These are the only two **fully present, real** shower datasets locally. `LEMURS_ODD_*.h5` is an
  empty file; `FCCeeALLEGRO` is present as 10 training-part files (~1.1M showers) plus small
  labeled test files at fixed `(E, φ, θ)` combos — this notebook uses **part1 only**
  (~100k showers, roughly matching CaloChallenge Dataset 2's 80k/20k train/test scale). Loading
  all 10 parts is a one-line change (`lemurs_data_path`) if more data is wanted later.
- They conveniently share the same `(R, PHI, Z) = (9, 16, 45)` axis convention (confirmed via
  `../LEMURS.ipynb`'s existing loader), so the same transpose-to-`(45,16,9)` used for CaloChallenge
  works unchanged for LEMURS — no re-derivation of the grid/patch logic needed.
- CaloChallenge Dataset 2 has no incident-angle field (fixed perpendicular incidence) — it gets a
  constant `(φ=0, θ=π/2)`, which not coincidentally sits at the center of FCCeeALLEGRO's own
  angular range (`θ∈[0.87, 2.27]`, `φ∈[-π, π]`), so the fixed value isn't an extrapolation case for
  the angle embedding.
- **Physical-unit RoPE / geometry-aware attention masking** (top-level README's "Architectural
  refinements" / "Actionable recipe" items 1, 4) are still **not** included — they normalize
  positional encoding across detectors with different physical dimensions per voxel, which is a
  reasonable next step now that a second real geometry exists, but is a bigger lift than what was
  needed to get joint training itself working.

## Data pipeline — `multi_geometry_dataloaders()` (in `../utilities.py`)

Combines both geometries into one `ConcatDataset`, each with its **own** fitted log-normalization
min/max (different absorbers/sampling fractions mean different deposit-energy scales — a shared
normalization would over/under-saturate one of them):

- CaloChallenge Dataset 2: reuses the existing `CaloChallenge` dataset class (HLF-based
  flat-vector → `(R,PHI,Z)` reshape), transposed to `(Z,PHI,R)`, tagged `geom_id=0`,
  `phi=0.0`, `theta=π/2` (constants).
- FCCeeALLEGRO: the `LEMURS` dataset class (in `../utilities.py`) was extended with optional
  `phi_key`/`theta_key` constructor args to load `incident_phi`/`incident_theta` per sample
  alongside `incident_energy` and `showers`; same `(R,PHI,Z)→(Z,PHI,R)` transpose, tagged
  `geom_id=1` with real per-sample angles.
- **Key-name gotcha**: `CaloChallenge` hardcodes `"incident_energies"` (plural) internally;
  LEMURS files use `"incident_energy"` (singular). `multi_geometry_dataloaders()` takes a
  dedicated `cfg["lemurs_energies_key"]` for this rather than reusing the shared `energies_key`.
- Each batch is `(x, cond, phi, theta, geom_id)`; `inverses` is a `{geom_id: inverse_fn}` dict
  since inversion depends on which geometry's min/max normalization a sample came from.

## Model — `ConditionalFlowMatchingDiT`

* **`PatchEmbed3D`** — `Conv3d(1, embed_dim, kernel=patch_size, stride=patch_size)` tokenizes the
  grid once. `patch_size=(5,4,3)` on `voxel_shape=(45,16,9)` → grid `(9,4,3)` = **108 tokens**.
* **`AxialRoPE3D`** — splits each attention head's dimension into 3 equal chunks (one per
  depth/angle/radius axis) and applies standard RoPE rotation to each using that axis's
  patch-grid coordinate. Requires `head_dim % 6 == 0`; with `embed_dim=384, num_heads=8` →
  `head_dim=48` → 16 rotary dims per axis.
* **`DiTBlock3D`** — pre-LN transformer block (self-attention + MLP), each sub-block modulated by
  adaLN-Zero (scale/shift before, gate after) from the merged conditioning embedding. Zero-init
  on the adaLN output projection makes every block the identity function at initialization
  (standard DiT stabilization trick) — as a consequence, gradients only start flowing back into
  the conditioning encoders (`energy_proj`, `angle_emb`, `geom_embed`) after a handful of
  optimizer steps move the adaLN weights off exact zero. Verified in a smoke test, not a bug.
* **Conditioning embedding** — four signals concatenated then merged by an MLP into one
  `embed_dim`-sized vector `c`, fed to every block's adaLN and the final layer's adaLN:
  - `TimeEmbedding(t·1000)` — continuous flow time
  - energy MLP on `log10(E_inc/GeV)`
  - **`AngleEmbedding`** — `MLP(sin φ, cos φ, sin θ, cos θ)`, avoiding the `φ=0/2π` discontinuity
    a raw-radian input would have
  - `geom_embed` lookup — `n_geometries=3`: `0`=SiW, `1`=FCCeeALLEGRO, `2`=**reserved** for a
    future third geometry (untouched by current data; its embedding row gets zero gradient, by
    construction, until something trains on `geom_id=2`)
* **`FinalLayer3D`** — adaLN-modulated LayerNorm + Linear projecting each token back to
  `patch_volume` values, unpatchified to `(B, 1, D, H, W)` via `einops.rearrange`. Zero-initialized
  so the network outputs `v≈0` at initialization.

## `RectifiedFlow` — training & sampling

* **Training** (`training_loss`): sample `t ~ U(0,1)`, `x0 ~ N(0,I)`, interpolate
  `x_t = (1-t)·x0 + t·x1`, regress `v_θ(x_t, t, cond, phi, theta, geom_id)` against the constant
  target velocity `x1 - x0` with plain MSE. This is the `σ_min=0` case of Lipman et al.'s
  conditional-OT path.
* **Sampling** (`sample`): integrate `dx/dt = v_θ(x, t, cond, phi, theta, geom_id)` from `t=0`
  (Gaussian noise) to `t=1` (data), via **Euler** (1 model call/step) or **Heun/RK2** (2 calls/step,
  better accuracy at low step counts — the default, `sampling_method="heun"`).
  `sampling_steps=50` by default; worth sweeping down (e.g. 10-20 steps) once trained, since flow
  matching typically needs far fewer NFE than DDIM's 1000-step training horizon.

## Pseudocode

```
# config: patch_size=(5,4,3) on voxel_shape=(45,16,9) -> grid (9,4,3) = 108 tokens
# embed_dim=384, depth=8, num_heads=8 (head_dim=48, 16 rotary dims/axis), n_geometries=3

function ConditionalFlowMatchingDiT.forward(x_t, t, cond, phi, theta, geom_id):
    tokens, grid_shape = PatchEmbed3D(x_t)          # (B, 1, D, H, W) -> (B, N=108, embed_dim)

    t_emb = TimeEmbedding(t * 1000)                  # continuous flow time -> embed_dim
    e_emb = MLP(log10(E_inc))                        # energy conditioning -> embed_dim
    a_emb = MLP(sin(phi), cos(phi), sin(theta), cos(theta))   # angle conditioning -> embed_dim
    g_emb = Embedding(geom_id)                        # 0=SiW, 1=FCCeeALLEGRO, 2=reserved
    c     = MLP(concat(t_emb, e_emb, a_emb, g_emb))  # merged conditioning vector, used everywhere below

    for block in DiTBlocks (x8):                      # global self-attention throughout, no downsampling
        shift1, scale1, gate1, shift2, scale2, gate2 = MLP_zero_init(c)   # adaLN-Zero
        h      = modulate(LayerNorm(tokens), scale1, shift1)
        tokens = tokens + gate1 * Attention3DRoPE(h)  # RoPE keyed on (d,h,w) patch coordinate
        h      = modulate(LayerNorm(tokens), scale2, shift2)
        tokens = tokens + gate2 * MLP(h)

    v_pred = FinalLayer3D(tokens, c, grid_shape)      # adaLN + Linear + unpatchify -> (B, 1, D, H, W)
    return v_pred                                     # predicted velocity x1 - x0

function RectifiedFlow.training_loss(model, x1, cond, phi, theta, geom_id):
    t  = uniform(0, 1)
    x0 = randn_like(x1)
    x_t = (1 - t) * x0 + t * x1
    return mse(model(x_t, t, cond, phi, theta, geom_id), x1 - x0)

function RectifiedFlow.sample(model, shape, cond, phi, theta, geom_id, steps=50, method="heun"):
    x = randn(shape)                                  # x_0 ~ N(0, I)
    for t_cur, t_next in linspace(0, 1, steps+1) pairs:
        v1 = model(x, t_cur, cond, phi, theta, geom_id)
        if method == "euler":
            x = x + (t_next - t_cur) * v1
        else:  # heun / RK2
            v2 = model(x + (t_next - t_cur) * v1, t_next, cond, phi, theta, geom_id)
            x  = x + (t_next - t_cur) * 0.5 * (v1 + v2)
    return x                                           # x_1 ~ data
```

## Train

`train()` mirrors the original DDIM notebook's loop (AdamW, cosine LR, grad clipping, AMP,
periodic checkpointing, resume via `cfg["scratch"]`, optional wandb logging); the per-batch unpack
is now `(x1, cond, phi, theta, geom_id)` from the combined multi-geometry loader.

`load_fm_checkpoint()` / `generate_fm()` are small **local** equivalents of
`../utilities.py`'s `load_model_from_checkpoint()` / `generate()` — not added to the shared
module, since those are hard-wired to `UNet3D`/`DDIMScheduler` kwargs that don't apply here.
`generate_fm()` takes `(e_inc_gev, phi, theta, geom_id)` with defaults matching CaloChallenge's
fixed incidence (`phi=0.0, theta=π/2, geom_id=0`); pass real angles + `geom_id=1` +
`inverse=inverses[1]` for FCCeeALLEGRO.

No checkpoint exists yet in `checkpoints/` — run the (commented-out) `train(CFG, train_loader)`
cell first before uncommenting the "Load pretrained model" / Eval cells.

## Eval

Two independent sweeps, one per geometry:

- **Geometry 0 (SiW)**: energy sweep over `E_INC ∈ {1, 10, 100, 1000, 2000} GeV` at the fixed
  angle, ground truth from `calo_test_data_path`. Same Z-profile / `plot_comparison()`
  (EMD/Wasserstein) plots as the original DDIM notebook.
- **Geometry 1 (FCCeeALLEGRO)**: sweeps the 8 labeled test files in
  `lemurs_test_dir` — every combination of `E∈{5,50} GeV`, `φ∈{0.0,0.2}`, `θ∈{1.57,2.1}` — parsed
  from filenames via regex, generated at the matching `(E,φ,θ)`, and compared against that file's
  own 1000 ground-truth showers. This is the eval that actually exercises whether angle
  conditioning learned something, since Dataset 2 alone can't test that (single fixed angle).

Both were run end-to-end (tiny model, `n_samples` capped, 2 epochs) against the real HDF5 files as
a wiring smoke test before this was handed off — training to convergence and reading the resulting
plots is the next step.

## Open questions worth revisiting once trained

- **Does angle conditioning actually work?** The FCCeeALLEGRO 8-combo sweep is the first real test
  of that — worth checking whether Z-profiles/radial profiles track the ground truth's `(φ,θ)`-
  dependent shape changes, or whether the model collapsed to an angle-averaged shower.
- **Does joint training help or hurt either geometry alone?** Worth comparing against a
  single-geometry ablation (e.g. re-run with only `geom_id=0` batches) to see whether sharing
  weights across SiW and FCCeeALLEGRO is a net positive, consistent with AllShowers' finding that
  joint multi-geometry training scales but doesn't automatically improve single-geometry quality.
- **Sampling steps**: sweep `sampling_steps` down (10-20) and compare EMD/FPD against this run.
- **More LEMURS data**: only `part1` (~100k showers) is used; the other 9 parts are unused unless
  `lemurs_data_path` is changed or extended to a multi-file loader.
- **Reserved geometry slot (`geom_id=2`)**: still a documented no-op — exercising it means adding
  a third real geometry (a re-fetched LEMURS ODD, or a re-binned Dataset 2).
