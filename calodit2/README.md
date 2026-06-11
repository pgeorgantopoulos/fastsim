# CaloDiT-2

Based on the details from the paper ***"A Generalisable Generative Model for Multi-Detector Calorimeter Simulation" (arXiv:2509.07700)***, here is the breakdown of the model's architecture, underlying PyTorch components, and hyperparameters:

### 1. Architecture & Underlying PyTorch Models

**CaloDiT-2** is an advanced generative model designed for fast calorimeter particle shower simulations in high-energy physics, integrated directly into the Geant4 simulation toolkit:

* **Diffusion Backbone:** It utilizes a **Diffusion Transformer (DiT)**-like architecture. Instead of discrete-time diffusion (DDPM), CaloDiT-2 shifts to the continuous-time **EDM** *(Elucidating the Design Space of Diffusion-Based Generative Models)* formulation, which frames the reverse process via a Probability Flow Ordinary Differential Equation (ODE).
* **Distillation Framework:** It incorporates **Consistency Distillation (CD)**. During training, a student model is trained from a pretrained EDM teacher model to bypass multi-step ODE solving, shrinking inference down to a single evaluation step.
* **Modality Adaptation:** Showers are tokenized via a large-patch representation (similar to ViTs). To support multiple detectors, it models universal 3D cylindrical grids ($N \times R \times \phi$) using local radiation lengths and Molière radii, rendering the data representation detector-agnostic.

### 2. Model Hyperparameters

The model uses a lightweight yet scalable transformer backbone config:

* **Transformer Layers (Depth):** 6 layers
* **Hidden Dimension:** 384
* **Grid/Voxel Size (CaloChallenge Dataset 2 standard):** $45 \times 9 \times 16 = 6,480$ total voxels.
* **Conditioning Dimensions:** * Incident particle properties (Kinetic Energy $E$, global incident angles $\phi$, $\theta$).
* A categorical one-hot geometry vector $G$ of size $K + 1$ (where $K$ represents pretrained detector geometries, plus an extra slot to allow zero-shot/few-shot adaptation to a novel unseen detector layout).


### 3. Training Hyperparameters

* **Optimization / Fine-Tuning Learning Rate:** Standard pre-training is performed across broad datasets (such as the 5-million shower LEMURS dataset). For downstream adaptation (transfer learning) onto a newly introduced detector, the learning rate is aggressively constrained to $\le 1 \times 10^{-3}$.
* **Data & Step Efficiency:** Due to the pre-training strategy, adapting the model to novel geometries requires up to **25× less data** and **20× fewer training steps** compared to training from scratch.
* **Infrastructure Framework:** The PyTorch codebase relies on Hugging Face `Accelerate` for distributed multi-GPU training, tracking progress via Weights & Biases (`wandb`). The final architectures are exported via `LibTorch` and `ONNX` to accommodate C++ inference wrappers within production simulation loops.

### 4. Detailed Technical Specifications (from paper)

**Data Preprocessing:**

Voxels below 15.1 keV are clipped to zero. Each voxel energy is then log-normalised using dataset-wide statistics:

$$\hat{x}_i = \frac{\log(x_i + \varepsilon) - \mu}{\sigma}, \quad \varepsilon = 10^{-6}$$

where $\mu$ and $\sigma$ are computed over the full training set. Conditioning variables are normalised as $\hat{E} = E/E_{\max}$, $\hat{\theta} = \theta/\pi$, $\hat{\phi} = [\sin\phi,\, \cos\phi]$.

**Conditioning Injection:**

Incident particle properties $(E, \phi, \theta)$ and diffusion timestep $t$ are passed through nonlinear projections and concatenated into a single condition embedding $c$, which is then fed to **every DiT block**. The geometry one-hot vector $G$ ($K{+}1$ classes) is similarly projected and concatenated into $c$.

**Patch Tokenization:**

* Patch size: **3 × 2 × 3** (z × φ × r)
* Number of tokens: **360**
* Positional encoding: **3D sinusoidal**, with the r, φ, and z directions each occupying one-third of the embedding dimension.

**EDM Preconditioning** (Karras et al. 2022):

$$s_\phi(x,t) = c_{\text{skip}}(t)\,x + c_{\text{out}}(t)\,v_\theta\!\left(c_{\text{in}}(t)\,x,\; c_{\text{noise}}(t)\right)$$

$$c_{\text{skip}} = \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + t^2}, \quad c_{\text{in}} = \frac{t\,\sigma_{\text{data}}}{\sqrt{\sigma_{\text{data}}^2 + t^2}}, \quad c_{\text{out}} = \frac{1}{\sqrt{\sigma_{\text{data}}^2 + t^2}}, \quad c_{\text{noise}} = \frac{\kappa}{4}\ln t$$

with $\sigma_{\text{data}} = 0.5$, $\sigma_{\min} = 0.002$, $\sigma_{\max} = 80$, $\kappa = 10^4$.

Loss: $\mathcal{L}_{\text{EDM}} = \mathbb{E}\!\left[\lambda(t)\,\|s_\phi(x,t) - x_0\|_2^2\right]$, $\lambda(t) = (t^2 + \sigma_{\text{data}}) / (t\,\sigma_{\text{data}})$.

**Consistency Distillation:**

$$\mathcal{L}_{\text{CD}} = \mathbb{E}\!\left[\lambda(t_n)\,\|f_\zeta(x_{t_{n+1}}, t_{n+1}) - f_{\zeta^-}(x_{t_n}, t_n)\|_2^2\right]$$

* Continuous time $[\varepsilon, T]$ discretised into **32 sub-intervals**
* $\lambda(t_n) = 1$; target network $\zeta^-$ updated via EMA with decay $\mu = 0.95$
* Student initialised from teacher weights; trained for **100K steps** with AdamW, lr = 5×10⁻⁴


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
| Patch tokenization | `PatchEmbed3D` is implemented but **not yet wired** into `UNet3D.forward` — it remains a conceptual sketch | Patch size **3×2×3** → **360 tokens**; 3D sinusoidal positional encoding (r/φ/z each occupy 1/3 of embedding space) |
| Depth / width | `base_ch=8`, `ch_mults=(1,2,4,8)` in the saved checkpoint | 6 transformer layers, hidden dim 384 |
| Inductive bias | Strong spatial locality bias from Conv3d, local skip connections | Minimal inductive bias — long-range dependencies from token attention |
| Inference export | Plain PyTorch | LibTorch + ONNX for C++ production integration in Geant4 |

The transformer blocks (`PatchEmbed3D` and `TransformerBlock`) are defined and documented in the notebook but disconnected from the `UNet3D` forward pass — the trained checkpoint still uses the pure convolutional UNet.

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

The notebook is a clean, complete DDIM baseline on the same LEMURS voxel grid, with a solid U-Net backbone and a well-structured extension point for transformers. CaloDiT-2 goes in three orthogonal directions:

1. **Architecture** — replaces the U-Net with a full DiT (patch tokens, global attention everywhere)
2. **Diffusion** — moves from discrete DDPM/DDIM to continuous EDM, then distills to single-step consistency
3. **Generalization** — adds multi-detector conditioning so one model covers multiple geometries with few-shot transfer

The next natural step would be to complete the wiring of `PatchEmbed3D` + `TransformerBlock` into the forward pass, replacing or augmenting the bottleneck first (lowest spatial resolution, most global features), which mirrors how CaloDiT-2 achieves full-sequence attention without the U-Net's per-voxel convolution cost.
