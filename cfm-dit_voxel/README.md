# Conditional Flow Matching + DiT, trained on two real geometries, tested zero-shot on a third

Conditional Flow Matching (Lipman et al., 2023, https://arxiv.org/abs/2210.02747; rectified-flow
path per Liu et al., 2022, https://arxiv.org/abs/2209.03003) applied to 3D voxelized calorimeter
shower data, with a pure Diffusion Transformer backbone, **jointly trained on two real geometries
and evaluated zero-shot on a third, held-out geometry** — mirroring CaloDiT-2/LEMURS's
held-out-detector methodology (arXiv:2509.07700):

- **`geom_id=0`** — CaloChallenge Dataset 2 (SiW), fixed incidence (`θ=π/2, φ=0`) — **trained**
- **`geom_id=1`** — LEMURS Par04SiW, variable incidence angle `(φ, θ)` per sample — **trained**
- **`geom_id=2`** — LEMURS FCCeeALLEGRO — **held out**: never appears in the training/test
  loaders, no gradient ever touches it. Only its own log-normalisation min/max is fit (so
  generated samples can be mapped back to MeV); its `geom_embed` row stays at random init
  throughout training. Used purely as a zero-shot generalisation eval.

FCCeeALLEGRO is held out rather than trained on because it's the most physically different of the
three available geometries (liquid-argon-lead, turbine-layout layers, vs. SiW/Par04SiW's more
similar Si-based concentric-cylinder designs) — the same reasoning CaloDiT-2/LEMURS used to pick
it as their held-out detector.

## Code layout

- **`cfm_dit_voxel.py`** — all model code: `ConditionalFlowMatchingDiT` (DiT backbone),
  `RectifiedFlow` (training loss + ODE sampler), and the `train`/`load_fm_checkpoint`/
  `generate_fm` helpers. Import from here rather than redefining anything inline.
- **`cfm-dit_voxel.ipynb`** — config (`CFG`), data loading, orchestration, eval plots. Imports
  everything model-related from `cfm_dit_voxel.py`.
- **`../utilities.py`** — shared dataset classes (`CaloChallenge`, `LEMURS`) and
  `multi_geometry_dataloaders()`, used by this notebook and others in the repo.

## Data — `multi_geometry_dataloaders()` (in `../utilities.py`)

Combines the two **trained** geometries into one `ConcatDataset`, each with its **own** fitted
log-normalization min/max (different absorbers/sampling fractions mean different deposit-energy
scales — a shared normalization would over/under-saturate one of them). Each batch is
`(x, cond, phi, theta, geom_id)` for `geom_id ∈ {0, 1}` only. `inverses` is a `{geom_id: fn}` dict
mapping `[-1,1]` back to MeV, keyed by which geometry's normalization a sample came from.

If `cfg["heldout_data_path"]` is set (LEMURS FCCeeALLEGRO here), the loader also fits *just* that
geometry's log-norm range and adds `inverses[2]` — without adding it to `train_ds`/`test_ds`. No
gradient ever touches it; it exists purely so eval cells can map `geom_id=2` generations back to
MeV for the zero-shot check.

Both trained geometries share the same `(R, PHI, Z) = (9, 16, 45)` axis convention, so the same
transpose-to-`(45,16,9)` used for CaloChallenge works unchanged for LEMURS. CaloChallenge Dataset 2
has no incident-angle field (fixed perpendicular incidence), so it gets a constant
`(φ=0, θ=π/2)` — which sits at the center of both LEMURS geometries' angular range, so it isn't an
extrapolation case for the angle embedding.

### Train/test split

`CaloChallenge.from_config`/`LEMURS.from_config` split each source file 80/20 by a plain
**sequential slice** (`idcs[:train_size]` / `idcs[train_size:]`, not a random shuffle) before
`multi_geometry_dataloaders()` concatenates the two trained geometries' train (resp. test) splits
into one `ConcatDataset`:

| geom_id | Geometry | File | Total | Train / Test (80/20) | Energy | Angle |
|---|---|---|---|---|---|---|
| 0 | CaloChallenge Ds2 (SiW, e⁻) | `calo_train_data_path` (`dataset_2_1.hdf5`) | 100,000 | 80,000 / 20,000 | 1 GeV – 1 TeV | fixed (θ=π/2, φ=0) |
| 1 | LEMURS Par04SiW (γ) | `lemurs_data_path` (`..._part1.h5`) | 100,000 | 80,000 / 20,000 | ~1 GeV – 1 TeV | variable, per-sample |
| 2 | LEMURS FCCeeALLEGRO (γ) — held out | `heldout_data_path` (`..._part1.h5`) | 99,942 | **0 / 0** (never in loaders) | 1 – 100 GeV | variable, per-sample |

#### Samples per energy decade

| geom_id | Geometry | 1–10 GeV | 10–100 GeV | 100–1000 GeV | Total |
|---|---|---|---|---|---|
| 0 | CaloChallenge Ds2 (SiW) | 33,476 (26,709 / 6,767) | 33,235 (26,591 / 6,644) | 33,289 (26,700 / 6,589) | 100,000 |
| 1 | LEMURS Par04SiW | 927 (748 / 179) | 8,963 (7,052 / 1,911) | 90,110 (72,200 / 17,910) | 100,000 |
| 2 | LEMURS FCCeeALLEGRO (held out) | 9,044 | 90,898 | *(range ends at 100 GeV)* | 99,942 |

Counts are `total (train / test)`; geom_id=2 has no train/test split (see above), so only a total is
shown. **SiW is sampled log-uniformly** — within half a point of 33% per decade, in both the full
file and each split. **Par04SiW and FCCeeALLEGRO are instead sampled uniformly in *linear* energy**:
for a `U(1, 1000)` GeV draw, the model `(x−1)/999` predicts 0.9% / 9.0% / 90.1% per decade, matching
the measured 0.93% / 8.96% / 90.11% almost exactly — so >90% of each LEMURS file sits in its top
decade and under 1% is below 10 GeV. The sequential 80/20 slice doesn't add any skew beyond what's
already in the source file order (train/test percentages agree with the full-file percentages to
within ~1pp in every decade, for both trained geometries).

#### Samples per incident angle (geom_id=0 is fixed, omitted)

| geom_id | Geometry | θ 0–10° off-axis | 10–20° | 20–30° | 30–40° | Total |
|---|---|---|---|---|---|---|
| 1 | LEMURS Par04SiW | 26,810 (21,408 / 5,402) | 26,014 (20,880 / 5,134) | 24,665 (19,771 / 4,894) | 22,511 (17,941 / 4,570) | 100,000 |
| 2 | LEMURS FCCeeALLEGRO (held out) | 26,671 | 26,145 | 24,765 | 22,361 | 99,942 |

Off-axis = `|θ − 90°|` (both files span `θ ∈ [49.9°, 130.2°]`, i.e. ±40° around CaloChallenge's fixed
perpendicular incidence). The slight taper toward ±30–40° is the expected shape for a solid-angle-
uniform generator (density ∝ sin θ falls off away from 90°). `φ` is uniform across all four quadrants
for both files (24.8–25.2% each — no meaningful skew) so it isn't broken out here. Par04SiW (trained)
and FCCeeALLEGRO (held out) use the same angular generation scheme, so the angle distribution the
model sees at train time for `geom_id=1` matches what it's asked to generalise to at `geom_id=2`.

Two more files are read only by eval cells, never by the train/test loaders themselves:
- `calo_test_data_path` (`dataset_2_2.hdf5`, 100,000 showers, same energy range) — ground truth for
  the geom_id=0 energy sweep.
- `lemurs_test_dir` (`FCCeeALLEGRO/..._1000events_5GeV_phi0.0_theta1.57.h5`, 1,000 showers) — one
  fixed-`(E,φ,θ)` labeled test file for the geom_id=2 zero-shot check; only 1 of the 8 such files in
  the Zenodo record's `testing_dataset_all_detectors.zip` is downloaded locally.

Only `part1` of LEMURS Par04SiW is used for training (9 more parts exist in the same Zenodo
record — see Known issues). The 80/20 split being a sequential slice rather than shuffled/
stratified turns out not to matter here — the per-decade breakdown above shows each split tracking
its full file's energy composition to within ~1pp, for both trained geometries.

## Model — `ConditionalFlowMatchingDiT`

* **`PatchEmbed3D`** — `Conv3d(1, embed_dim, kernel=patch_size, stride=patch_size)` tokenizes the
  grid once. `patch_size=(5,4,3)` on `voxel_shape=(45,16,9)` → grid `(9,4,3)` = **108 tokens**.
* **`AxialRoPE3D`** — splits each attention head's dimension into 3 equal chunks (depth/angle/
  radius) and applies RoPE rotation to each using that axis's patch-grid coordinate. Requires
  `head_dim % 6 == 0`; with `embed_dim=384, num_heads=8` → `head_dim=48` → 16 rotary dims/axis.
* **`RMSNorm` (QK-norm)** — normalizes q/k per head before the RoPE rotation, keeping attention
  logits bounded regardless of upstream activation scale. Needed to stop a training instability —
  see **Training** below.
* **`DiTBlock3D`** — pre-LN transformer block (self-attention + MLP), each sub-block modulated by
  adaLN-Zero (scale/shift before, gate after) from the merged conditioning embedding. Zero-init on
  the adaLN output projection makes every block the identity function at initialization.
* **Conditioning embedding** — four signals concatenated then merged by an MLP into one
  `embed_dim`-sized vector `c`, fed to every block's adaLN and the final layer's adaLN:
  `TimeEmbedding(t·1000)` (flow time), an energy MLP on `log10(E_inc/GeV)`, `AngleEmbedding`
  (`MLP(sin φ, cos φ, sin θ, cos θ)`, avoiding the `φ=0/2π` discontinuity), and a `geom_embed`
  lookup (`n_geometries=3`).
* **`FinalLayer3D`** — adaLN-modulated LayerNorm + Linear projecting each token back to
  `patch_volume` values, unpatchified via `einops.rearrange`. Zero-initialized so the network
  outputs `v≈0` at initialization.

## `RectifiedFlow` — training & sampling

* **Training**: sample `t ~ U(0,1)`, `x0 ~ N(0,I)`, interpolate `x_t = (1-t)·x0 + t·x1`, regress
  `v_θ(x_t, t, cond, phi, theta, geom_id)` against the constant target velocity `x1 - x0` with MSE
  (the `σ_min=0` case of Lipman et al.'s conditional-OT path).
* **Sampling**: integrate `dx/dt = v_θ(...)` from `t=0` (noise) to `t=1` (data), via Euler (1
  model call/step) or Heun/RK2 (2 calls/step, default, `sampling_method="heun"`). `x` is clamped
  to `[-1,1]` after every step (not just the final one) — uncorrected mid-integration excursions
  pass through the log-normalisation's `exp()` inverse and turn into large fake MeV spikes.

```
function ConditionalFlowMatchingDiT.forward(x_t, t, cond, phi, theta, geom_id):
    tokens, grid_shape = PatchEmbed3D(x_t)             # (B, 1, D, H, W) -> (B, N=108, embed_dim)

    t_emb = TimeEmbedding(t * 1000)                     # continuous flow time -> embed_dim
    e_emb = MLP(log10(E_inc))                           # energy conditioning -> embed_dim
    a_emb = MLP(sin(phi), cos(phi), sin(theta), cos(theta))
    g_emb = Embedding(geom_id)                          # 0=SiW, 1=Par04SiW, 2=held out
    c     = MLP(concat(t_emb, e_emb, a_emb, g_emb))     # merged conditioning vector

    for block in DiTBlocks (x8):                        # global self-attention, no downsampling
        shift1, scale1, gate1, shift2, scale2, gate2 = MLP_zero_init(c)   # adaLN-Zero
        h      = modulate(LayerNorm(tokens), scale1, shift1)
        tokens = tokens + gate1 * Attention3DRoPE(QKNorm(h))   # RoPE keyed on (d,h,w) coordinate
        h      = modulate(LayerNorm(tokens), scale2, shift2)
        tokens = tokens + gate2 * MLP(h)

    return FinalLayer3D(tokens, c, grid_shape)          # adaLN + Linear + unpatchify

function RectifiedFlow.sample(model, shape, cond, phi, theta, geom_id, steps=50, method="heun"):
    x = randn(shape)
    for t_cur, t_next in linspace(0, 1, steps+1) pairs:
        v1 = model(x, t_cur, cond, phi, theta, geom_id)
        x  = x + (t_next - t_cur) * v1                              if method == "euler" else
        x  = x + (t_next - t_cur) * 0.5 * (v1 + model(x + (t_next - t_cur) * v1, t_next, ...))
        x  = clamp(x, -1, 1)
    return x
```

## Training

Toggle `CFG["mode"]`: `"train"` runs `train()` fresh (or resumed if `CFG["scratch"]=0`); `"eval"`
loads the latest checkpoint from `CFG["ckpt_dir"]` and skips training.

**Current status: trains cleanly**, converging smoothly with grad norms staying bounded
throughout. QK-norm (`RMSNorm` on q/k before RoPE, in `Attention3DRoPE`) is what keeps
attention logits — and hence gradients — from growing unbounded over a long schedule; the
`grad_norm_ok` guard in `train()` (reject, don't just clip, any step whose pre-clip norm
exceeds `grad_clip * grad_skip_mult`) is a safety net on top of that, calibrated to this
architecture's normal grad-norm scale (see `CFG["grad_skip_mult"]`'s comment).

## Eval

- **Geometry 0 (SiW)**: energy sweep over `E_INC ∈ {1, 10, 100, 1000, 2000} GeV` at the fixed
  angle, ground truth from `calo_test_data_path`. Z-profile plots + `plot_comparison()`
  (EMD/Wasserstein).
- **Geometry 2 (FCCeeALLEGRO, held out)**: sweeps the labeled test files in `lemurs_test_dir`,
  generated at the matching `(E,φ,θ)` with `geom_id=2`, compared against that file's ground truth.
  Only 1 of 8 fixed-`(E,φ,θ)` test files is downloaded locally right now
  (`..._5GeV_phi0.0_theta1.57.h5`); the rest ship in the same Zenodo record's
  (`10.5281/zenodo.17045562`) `testing_dataset_all_detectors.zip`.
- **Geometry 1 (Par04SiW) has no dedicated eval cell yet** — it only contributes to the combined
  `test_loss` logged during training.

## Known issues / next steps

- **Geometry 0 (SiW) shower-quality re-verification**: Z-profiles/EMD against Geant4 should be
  re-checked against the current checkpoint — worth watching in particular whether sparse-voxel
  log-normalization sensitivity resurfaces (the log+min/max normalization squeezes a wide MeV
  range into `[-1,1]`, and with ~76% of voxels exactly zero, a small residual model error near the
  "empty voxel" boundary can blow up multiplicatively after the inverse transform). If it does, try
  a larger `log_eps` or a post-hoc MeV floor on generated output.
- **Zero-shot generalisation to FCCeeALLEGRO** needs the eval cells run against the current
  checkpoint to see whether Z-profiles land anywhere near ground truth, and whether angle
  conditioning (learned from Par04SiW's variable angles) transfers.
- **Joint vs. single-geometry training**: no ablation yet comparing joint SiW+Par04SiW training
  against training on SiW alone.
- **Sampling steps**: `sampling_steps=50` (Heun) is untuned — worth sweeping down (10-20) and
  comparing EMD once eval is re-run.
- **Par04SiW eval coverage**: no dedicated energy-sweep/Z-profile section yet (see Eval above).
- **More LEMURS training data**: only Par04SiW's `part1` (~100k showers) is used; 9 more parts are
  available in the same Zenodo record.
