# DDIM Training on LEMURS 3D Calorimeter Data

Denoising Diffusion Implicit Models (Song et al., 2020, https://arxiv.org/pdf/2010.02502) applied to 3D voxelized
calorimeter shower data from the CERN LEMURS dataset.

Training objective is identical to DDPM (predict ε), but sampling is non-Markovian,
allowing orders-of-magnitude fewer inference steps.
**CFG dict** — single config object covering data path, voxel geometry (45, 16, 9) matching LEMURS (N_layers, N_alpha, N_r), model width, diffusion schedule, and training hyperparameters.

## Dataset

`CaloChallenge` (in `../utilities.py`) reads the HDF5 file and reshapes each flat voxel
row into `(R, PHI, Z)` using the layer/angle/radius bin edges from `HighLevelFeatures`
(binning XML). `ddim_calochallenge_dataloaders()` wraps it with a `transform`:

* reshapes each shower from `(R, PHI, Z)` to `(Z, PHI, R)` = `voxel_shape` order
* applies log-normalisation `log(E + ε)` to handle the sparse, heavy-tailed energy
  distribution, then min/max-scales to `[-1, 1]` — min/max are fit on the training
  split only (as in <https://arxiv.org/pdf/2308.03876>, Eq.8)
* returns `(x, cond)` where `cond = log10(E_inc_MeV + ε)`
* also returns an `inverse()` closure that maps a `[-1, 1]` sample back to raw MeV deposits, used at generation time

## UNet3D+Transformer — 3-D AutoEncoder:

* Sinusoidal time embedding fed through an MLP
* Incident-energy conditioned via a separate projection head
* FiLM-style (scale + shift) conditioning in every ResBlock3D
* SelfAttn3D at the coarsest spatial resolution
* Symmetric decoder with skip connections and F.interpolate to handle odd spatial dims

To integrate transformer blocks into the UNet3D, you would typically follow a strategy similar to what's done in Vision Transformers (ViT) or TransUNet (<https://arxiv.org/pdf/2102.04306>). The core idea is to replace or augment certain convolutional parts of the UNet with transformer layers, allowing the model to capture long-range dependencies more effectively.

1.  **Patch Embedding**: Instead of raw voxel data, the input to the transformer blocks needs to be a sequence of tokens. This is achieved by dividing the 3D voxel input into non-overlapping or overlapping 3D patches and then linearly projecting each patch into a higher-dimensional embedding space. This effectively flattens spatial information into a sequence.

2.  **Positional Encoding**: Since transformers are permutation-invariant (they don't inherently understand spatial relationships), you need to add positional embeddings to the patch embeddings. This injects spatial awareness back into the tokens.

3.  **Transformer Blocks**: These blocks consist of Multi-Head Self-Attention (MHSA) and a Feed-Forward Network (FFN), usually with Layer Normalization and residual connections. These blocks would operate on the sequence of patch embeddings.

4.  **Integration into UNet Architecture (Encoder/Decoder)**:
    *   **Encoder**: sequences of patch embeddings followed by transformer blocks. The patch embedding effectively handles the initial 'tokenization' and potentially downsampling. You'd need a mechanism to progressively reduce the sequence length or token dimensions.
    *   **Bottleneck**: transformer encoder, where the most abstract features are processed globally.
    *   **Decoder**: patch upsampling layer. Skip connections adapted to handle either token sequences or feature maps.

5.  **Conditioning**: The existing time and energy embeddings added to the patch embeddings with FiLM conditioning within the FFN of the transformer blocks.

**DDIMScheduler** — cosine or linear β schedule, q_sample for the forward pass, training_loss (plain MSE on noise), and ddim_sample with configurable steps and η (0 = deterministic, 1 = DDPM-equivalent).

train() — standard loop with gradient clipping, cosine-annealing LR, AMP (`torch.autocast` + `GradScaler`) on CUDA, periodic checkpointing (model/opt/sched/scaler state), resume-from-latest-checkpoint via `cfg["scratch"]`, and optional per-epoch wandb logging of loss/LR.

`load_model_from_checkpoint()` (in `../utilities.py`) loads the most recent checkpoint under `ckpt_dir` and rebuilds the model/scheduler from the `cfg` stored inside it, so eval always matches the architecture it was trained with.

## Pseudocode

```
# config: base_ch=16, ch_mults=(1,2,4,8) -> channels=[16,32,64,128], n_levels=4. Reduce ch_mults to (1, 2, 4) and base_ch to 16 if GPU memory is tight
# voxel_shape = (45,16,9); attn_res=(2,) -> attention only where min(spatial)//2^lvl == 2

function UNet3D.forward(x_t, t, cond):
    t_emb = TimeEmbedding(t)                 # sinusoidal -> MLP
    c_emb = MLP(log10(E_inc))                # energy conditioning
    emb   = concat(t_emb, c_emb)             # full conditioning vector, used everywhere below

    # ---------------- ENCODER ----------------
    h = Conv3d_in(x_t)                       # 1 -> channels[0]
    skips = []
    for level, out_ch in enumerate(channels):        # [16, 32, 64, 128] 
        h = ResBlock(h, emb)                 # FiLM-conditioned conv
        h = ResBlock(h, emb)
        if resolution_at(level) in attn_res:
            h = SelfAttn3D(h)                # full attention, only at res==2
        skips.append(h)                      # stash for decoder skip connection
        if level < n_levels - 1:
            h = Conv3d_stride2(h)            # downsample by 2x in D,H,W

    # ---------------- BOTTLENECK ----------------
    h = ResBlock(h, emb)
    tokens = PatchEmbed3D(h, patch_size=1)   # reshape voxels -> sequence of tokens
    tokens = TransformerBlock(tokens, emb)   # global self-attn + FiLM-conditioned MLP
    h = reshape(tokens -> 3D grid)
    h = ResBlock(h, emb)

    # ---------------- DECODER (mirrors encoder, reversed) ----------------
    for level, (out_ch, skip) in enumerate(zip(reversed(channels), reversed(skips))):
        if level > 0:
            h = ConvTranspose3d_stride2(h)   # upsample by 2x
        h = interpolate(h, size=skip.shape)  # match spatial dims exactly
        h = concat(h, skip, dim=channel)     # U-Net skip connection
        h = ResBlock(h, emb)
        h = ResBlock(h, emb)
        if resolution_at(level) in attn_res:
            h = SelfAttn3D(h)

    return Conv3d_out(SiLU(GroupNorm(h)))    # -> predicted noise eps, same shape as x_t
```

## Eval

After loading the checkpoint, the notebook runs an energy sweep and compares generated
showers against Geant4 ground truth from the test file:

* **Sweep generation** — for each `E_INC ∈ {1, 10, 100, 1000, 2000} GeV`, calls
  `generate()` (in `../utilities.py`) to draw `N_SWEEP=100` DDIM samples (50 steps,
  η=0) conditioned on that energy, and pulls matching ground-truth showers from
  `test_data_path` via a `±2x` energy-window mask.
* **Z-profiles** — per energy, plots per-shower energy deposit summed over (φ, r) vs.
  depth layer, generated overlaid on ground truth, to check longitudinal shower shape.
* **Energy distribution / radial profile comparison** — reshapes sweep data from
  `(N, Z, PHI, R)` to `(N, R, PHI, Z)` and calls `plot_comparison()` (in
  `../utilities.py`), which plots histograms with ratio panels and Earth Mover's
  Distance (Wasserstein) scores between generated and reference distributions.


# DDIM+T vs CaloDiT-2

## Model & Data Comparison: `ddim_transformer.ipynb` vs. CaloDiT-2

### Data Representation

| Aspect | `ddim_transformer.ipynb` | CaloDiT-2 |
|---|---|---|
| Dataset | LEMURS HDF5, single geometry (`SiW_gamma`) | LEMURS 5M-shower multi-geometry dataset |
| Voxel grid | `(45, 16, 9)` = 6,480 voxels, D×H×W | Same `45 × 9 × 16` standard, but with detector-agnostic normalization |
| Spatial axes | Raw integer bins (N_layers, N_alpha, N_r) | Physical units: local radiation lengths + Molière radii |
| Preprocessing | `log(E + ε)`, scaled to `[-1, 1]`; global min/max | Clip < 15.1 keV → 0, then `x̂ = (log(x + ε) - μ) / σ` with per-dataset μ, σ; ε = 1e-6 |
| Conditioning | One scalar: `log10(E_inc)` | `(E, φ, θ)` + one-hot geometry vector `G` of size `K+1` |
| Multi-detector | No — single fixed geometry | Core design goal; zero/few-shot transfer to new detectors |

The most significant gap is in conditioning. The notebook model receives a single energy scalar; CaloDiT-2 receives full incident kinematics plus a geometry token that makes the same model work across multiple detectors without retraining.

---

### Model Architecture

| Aspect | `ddim_transformer.ipynb` | CaloDiT-2 |
|---|---|---|
| Backbone | 3D U-Net (`UNet3D`) with convolutional encoder-decoder + skip connections | Pure Diffusion Transformer (DiT) — no convolutions |
| Attention | `SelfAttn3D` at the coarsest spatial level only (voxel-flattened tokens) | Global self-attention at all layers (token sequence throughout) |
| Conditioning mechanism | FiLM (scale + shift) in every `ResBlock3D` | Concatenated embedding `c` (from projected E, φ, θ, t, G) injected into every DiT block |
| Patch tokenization | `PatchEmbed3D` **is wired** into the `UNet3D` bottleneck (`patch_size=(1,1,1)`, one token per bottleneck voxel), followed by a single FiLM-conditioned `TransformerBlock` before reshaping back to a 3D grid | Patch size **3×2×3** → **360 tokens**; 3D sinusoidal positional encoding (r/φ/z each occupy 1/3 of embedding space) |
| Depth / width | `base_ch=16`, `ch_mults=(1,2,4,8)` → channels `[16,32,64,128]` | 6 transformer layers, hidden dim 384 |
| Inductive bias | Strong spatial locality bias from Conv3d, local skip connections | Minimal inductive bias — long-range dependencies from token attention |
| Inference export | Plain PyTorch | LibTorch + ONNX for C++ production integration in Geant4 |

---

### Diffusion Formulation

| Aspect | `ddim_transformer.ipynb` | CaloDiT-2 |
|---|---|---|
| Framework | Discrete-time DDPM/DDIM (T=1000) | Continuous-time EDM (Probability Flow ODE) |
| Training objective | MSE on predicted noise `ε` | EDM: `ℒ = E[λ(t)‖c_skip·x + c_out·vθ(c_in·x, c_noise) − x₀‖₂²]`; σ_data=0.5, σ_max=80 |
| Sampling | DDIM with configurable steps (e.g. 50) and `η` | CD: `ℒ_CD = E[λ(t_n)‖f_ζ(x_{t+1}, t+1) − f_ζ⁻(x_t, t)‖₂²]`; 32 sub-intervals, EMA μ=0.95 → **single-step** |
| Stochasticity | Tunable via `η`: 0 = deterministic, 1 = DDPM | Deterministic single-step student |
| NFE at inference | ~50 steps typical | 1 step (after distillation) |

This is the deepest architectural gap. DDIM cuts inference from 1000 to ~50 steps; CaloDiT-2's consistency distillation further compresses that to 1 step, which is what Geant4 integration actually needs for production speed.

---

### Summary 

CaloDiT-2 axes:

1. **Architecture** — replaces the U-Net with a full DiT (patch tokens, global attention everywhere)
2. **Diffusion** — moves from discrete DDPM/DDIM to continuous EDM, then distills to single-step consistency
3. **Generalization** — adds multi-detector conditioning so one model covers multiple geometries with few-shot transfer