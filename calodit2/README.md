# CaloDiT-2

Based on the details from the paper ***"A Generalisable Generative Model for Multi-Detector Calorimeter Simulation" (arXiv:2509.07700)***, here is the breakdown of the model's architecture, underlying PyTorch components, and hyperparameters:

## 1. Architecture & Underlying PyTorch Models

**CaloDiT-2** is an advanced generative model designed for fast calorimeter particle shower simulations in high-energy physics, integrated directly into the Geant4 simulation toolkit:

* **Diffusion Backbone:** It utilizes a **Diffusion Transformer (DiT)**-like architecture. Instead of discrete-time diffusion (DDPM), CaloDiT-2 shifts to the continuous-time **EDM** *(Elucidating the Design Space of Diffusion-Based Generative Models)* formulation, which frames the reverse process via a Probability Flow Ordinary Differential Equation (ODE).
* **Distillation Framework:** It incorporates **Consistency Distillation (CD)**. During training, a student model is trained from a pretrained EDM teacher model to bypass multi-step ODE solving, shrinking inference down to a single evaluation step.
* **Modality Adaptation:** Showers are tokenized via a large-patch representation (similar to ViTs). To support multiple detectors, it models universal 3D cylindrical grids ($N \times R \times \phi$) using local radiation lengths and Molière radii, rendering the data representation detector-agnostic.

## 2. Model Hyperparameters

The model uses a lightweight yet scalable transformer backbone config:

* **Transformer Layers (Depth):** 6 layers
* **Hidden Dimension:** 384
* **Grid/Voxel Size (CaloChallenge Dataset 2 standard):** $45 \times 9 \times 16 = 6,480$ total voxels.
* **Conditioning Dimensions:** * Incident particle properties (Kinetic Energy $E$, global incident angles $\phi$, $\theta$).
* A categorical one-hot geometry vector $G$ of size $K + 1$ (where $K$ represents pretrained detector geometries, plus an extra slot to allow zero-shot/few-shot adaptation to a novel unseen detector layout).


## 3. Training Hyperparameters

* **Optimization / Fine-Tuning Learning Rate:** Standard pre-training is performed across broad datasets (such as the 5-million shower LEMURS dataset). For downstream adaptation (transfer learning) onto a newly introduced detector, the learning rate is aggressively constrained to $\le 1 \times 10^{-3}$.
* **Data & Step Efficiency:** Due to the pre-training strategy, adapting the model to novel geometries requires up to **25× less data** and **20× fewer training steps** compared to training from scratch.
* **Infrastructure Framework:** The PyTorch codebase relies on Hugging Face `Accelerate` for distributed multi-GPU training, tracking progress via Weights & Biases (`wandb`). The final architectures are exported via `LibTorch` and `ONNX` to accommodate C++ inference wrappers within production simulation loops.

## 4. Detailed Technical Specifications (from paper)

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

## ALTERNATIVE Summary (Copilot)

Now let me get more details on the preprocessing and the full CaloDiT architecture:## Summary: NN Backbone, Preprocessing, and Data Flow in `validate.py`

### **1. NN Backbone: CaloDiT**

**Architecture:**
- **DiT-based (Diffusion Transformer)** inspired by [Facebook's DiT](https://github.com/facebookresearch/DiT)
- **Key Components:**
  - **Patch Embedding (VolumeEmbedder):** Converts 3D shower volume to patches and projects to embedding dimension
  - **Positional Embeddings:** 3D sinusoidal position encoding (frozen, not learned)
  - **Timestep Embedder:** Embeds diffusion timestep `t` via sinusoidal → 2-layer MLP
  - **Condition Embedders:** Multiple MLPs for embedding conditions (energy, phi, theta, geometry)
  - **Transformer Blocks (CaloDiTBlock):** `num_layers` blocks with:
    - Multi-head attention (adaptive Layer Norm modulation)
    - MLP (GELU activation)
    - Residual connections + gating
  - **Final Layer:** Reconstructs patches with adaptive normalization
  - **Unpatchify (VolumeUnembedder):** Unpacks patches back to 3D volume

**Config (from edm_allegro_scratch):**
```yaml
input_size: [9, 16, 45]        # (R, PHI, Z) dimensions
patch_size: [3, 2, 3]           # patch dimensions
conditions_size: [1, 2, 1]      # (energy, phi_2d, theta)
emb_dim: 384
num_layers: 6
num_heads: 6
mlp_ratio: 4
```

---

### **2. Data Preprocessing Pipeline**

**Shower Preprocessing (on raw calorimeter deposits):**
1. **CutNoise:** Remove energies below `noise_level=1.515e-05` GeV (energy readout threshold)
2. **LogTransform:** `log(x + eps)` with `eps=1e-6`
3. **Standardize:** `(x - mean) / std` with `mean=-10.766, std=3.5773`

**Condition Preprocessing (on incident particle parameters):**
- **Energy:** Normalized to [0, 1] by dividing by 1000 GeV max
- **Phi:** Converted from radians to 2D representation `[sin(φ), cos(φ)]`
- **Theta:** Normalized to [0, 1] using `(θ - θ_min) / (θ_max - θ_min)`
- **Geometry (optional):** One-hot encoding appended (for multi-geometry models)

---

### **3. Data Flow in `validate.py`**

```
┌─────────────────────────────────────────────────────────────────────┐
│ validate.py Data Flow                                               │
└─────────────────────────────────────────────────────────────────────┘

For each condition (geometry, energy, phi, theta):

1. LOAD & PREPARE
   └─ CaloShowerDataset(files=[fullsim_path], need_geo_condn)
      └─ Load reference showers from H5 file

2. BATCH & PREPROCESS (Training Loop)
   for sample in DataLoader:
      showers, conditions = sample  # Raw data
      │
      ├─ preprocessor.transform(conditions)
      │  └─ ConditionsPreprocessor: normalize energy/phi/theta
      │     └─ conditions_processed
      │
      ├─ cut_below_noise_level(showers)  # Remove noise
      │  └─ original_events (for comparison)
      │
      └─ FORWARD PASS:
         ├─ model.sample(conditions, progress=True, **cfg.sampling)
         │  ├─ Diffusion reverse process (iterative denoising)
         │  ├─ Calls CaloDiT iteratively with timestep conditioning
         │  └─ Returns generated showers (noisy → clean)
         │
         └─ preprocessor.inverse_transform(generated_events, conditions)
            └─ LogTransform⁻¹ → Standardize⁻¹
               └─ generated_events (physical scale)

3. COLLECT & SAVE
   original_events = concatenate all batches    → Geant4 reference
   generated_events = concatenate all batches   → CaloDiT output
   
   save_showers(generated_events, ...) → H5 file

4. EVALUATE
   compare_observables(original_events, generated_events)
   └─ Compute EMD distances on observables (energy, angular profiles)
```

---

### **Key Points:**

- **Backbone:** Vision Transformer adapted for 3D calorimeter data (patches + adaptive conditioning)
- **Preprocessing:** Noise cutting → log transform → standardization (reversible)
- **Conditioning:** Timestep + incident particle physics (E, φ, θ) + geometry
- **Forward Pass:** Diffusion sampling (not training) — iterative denoising with model predictions
- **Output:** Generated showers + comparison metrics (EMD)

### Training Data

From `configs/train/edm_allegro_scratch.yaml`:

**Training (parts 1–4):**
```
- ddsim_mesh_FCCeeALLEGRO_gamma_100kevents_1GeV100GeV_GPSFlat_edm4hep_13245648_part1.h5
- ddsim_mesh_FCCeeALLEGRO_gamma_100kevents_1GeV100GeV_GPSFlat_edm4hep_13245648_part2.h5
- ddsim_mesh_FCCeeALLEGRO_gamma_100kevents_1GeV100GeV_GPSFlat_edm4hep_13245648_part3.h5
- ddsim_mesh_FCCeeALLEGRO_gamma_100kevents_1GeV100GeV_GPSFlat_edm4hep_13245648_part4.h5
```
All located at: `/eos/geant4/fastSim/ddfastsim/FCCeeALLEGRO/dataset2_1GeV100GeVFlat_theta0p87to2p27_phiFull/`

**Validation (part 10):**
```
- ddsim_mesh_FCCeeALLEGRO_gamma_100kevents_1GeV100GeV_GPSFlat_edm4hep_13245648_part10.h5
```

### Testing/Validation Data

From `configs/validate/default.yaml`:

The validation script compares generated showers against **8 different datasets** (discrete, held-out from training):

| Geometry | Energy | File |
|----------|--------|------|
| Par04 | 50 GeV | `ddsim_mesh_Par04_gamma_1000events_50GeV_phi0.0_theta1.57_edm4hep_9246142.0.h5` |
| Par04 | 500 GeV | `ddsim_mesh_Par04_gamma_1000events_500GeV_phi0.0_theta1.57_edm4hep_9246143.0.h5` |
| SciPb | 50 GeV | `ddsim_mesh_Par04_gamma_1000events_50GeV_phi0.0_theta1.57_edm4hep_13208241.0.h5` |
| SciPb | 500 GeV | `ddsim_mesh_Par04_gamma_1000events_500GeV_phi0.0_theta1.57_edm4hep_13208242.0.h5` |
| ODD | 50 GeV | `ddsim_mesh_ODD_gamma_1000events_50GeV_phi0.0_theta1.57_edm4hep_9260460.0.h5` |
| ODD | 500 GeV | `ddsim_mesh_ODD_gamma_1000events_500GeV_phi0.0_theta1.57_edm4hep_9260457.0.h5` |
| CLD | 5 GeV | `ddsim_mesh_FCCeeCLD_gamma_1000events_5GeV_phi0.0_theta1.57_edm4hep_13208262.0.h5` |
| CLD | 50 GeV | `ddsim_mesh_FCCeeCLD_gamma_1000events_50GeV_phi0.0_theta1.57_edm4hep_9292534.0.h5` |

**Key difference:** Training uses only **FCCeeALLEGRO** (1 geometry), while validation tests across **4 geometries** (Par04, SciPb, ODD, CLD) at discrete energy points. The validation files are also completely separate from training—no overlap.

### Validation Metrics

1. Longitudinal Profile Observables

    LongTotalEnergy — Energy deposited per Z-layer
    LongTotalHits — Number of hits per Z-layer
    LongFirstMoment — Mean depth <z> distribution
    LongSecondMoment — Depth variance <z²> distribution
    LongEventEnergy — Per-layer energy distribution across events (9 layers)

2. Radial Profile Observables

    RadTotalEnergy — Energy deposited per radial bin (R-direction)
    RadTotalHits — Number of hits per radial bin
    RadFirstMoment — Mean radius <r> distribution
    RadSecondMoment — Radial variance <r²> distribution
    RadEventEnergy — Per-layer energy distribution (3×3 grid)

3. Azimuthal Profile Observables

    AzimTotalEnergy — Energy deposited per φ-bin
    AzimTotalHits — Number of hits per φ-bin
    AzimFirstMoment — Mean azimuthal angle <φ> distribution
    AzimSecondMoment — Azimuthal variance <φ²> distribution
    AzimEventEnergy — Per-layer energy distribution (4×4 grid)

4. Global Shower Observables

    TotalEventEnergy — Total energy per event distribution
    TotalEventHits — Total hits per event distribution
    CellEnergy — Individual cell energy distribution (linear scale)
    CellLogEnergy — Individual cell energy distribution (log₁₀ scale)
    CellEnergy_xlog — Cell energy with log-scale x-axis


