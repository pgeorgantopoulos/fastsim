# Point Cloud & Graph-Based Models for Fast Calorimeter Simulation

A literature collection on point cloud and graph-based generative models for fast calorimeter shower simulation (CaloChallenge, LEMURS, and other cell-hit level data), for comparison against voxel-based approaches.

## Calorimeter-specific point cloud models

- **[AllShowers](https://arxiv.org/pdf/2601.11716)** (Buss, Day-Hall, Gaede, Kasieczka, Krüger, 2026) — single continuous normalizing flow + Transformer model generating variable-length point clouds across *all* particle species (e/γ/hadrons) without retraining. Conditions on detector structure via **layer embeddings + Fourier position features**, plus **custom attention masking that enforces shower causality** (a voxel can only attend to layers a particle would have already traversed — earlier layers can't see later ones). Also trains jointly across multiple detector geometries (not just particle species), and introduces **shower/layer-wise optimal transport** as a training objective that compares generated vs. simulated distributions layer-by-layer instead of only globally. See the [geometry conditioning](#geometry-conditioning-why-and-how) section below for why this doesn't amount to true zero-shot generalization.
- **[CaloClouds](https://arxiv.org/abs/2305.04847)** → **[CaloClouds II](https://arxiv.org/pdf/2309.05704)** → **[CaloClouds3](https://arxiv.org/abs/2511.01460)** (DESY/FLC-QU group) — point cloud diffusion lineage. CaloClouds3 adds `ShowerFlow` (normalizing flow along the shower axis) + distilled diffusion for individual points, with angular conditioning for full-barrel coverage. [GitHub](https://github.com/FLC-QU-hep/CaloClouds).
- **[CaloPointFlow](https://ml4physicalsciences.github.io/2022/files/NeurIPS_ML4PS_2022_77.pdf)** → **[CaloPointFlow II](https://arxiv.org/abs/2403.15782)** — normalizing-flow point cloud model; v2 adds `CDF-Dequantization` and a `DeepSet-Flow` architecture to capture point-to-point correlations and exploit rotational symmetry (attacks the "multiple hit" collision problem). Benchmarked on CaloChallenge Dataset 2 and 3.

## Voxel / diffusion-transformer models

- **[CaloDiT-2](https://arxiv.org/html/2509.07700)** (`calodit2/`) — diffusion transformer with adaLN conditioning on (E, φ, θ) plus an explicit geometry one-hot vector; pretrained across 5 LEMURS geometries then fine-tuned per new detector. This is the main precedent for our geometry-conditioning design — full breakdown in the [geometry conditioning](#geometry-conditioning-why-and-how) section.
- **[CaloArt](https://arxiv.org/abs/2605.12011)** — large-patch, x-prediction diffusion transformer for high-granularity voxel grids. Best FPD / classifier metrics on CaloChallenge Dataset 2 and 3 at ~10ms/shower. Trains **one model per fixed detector** with no geometry-conditioning mechanism at all, but contributes a reusable architectural idea (3D axial RoPE position encoding) — see [architectural refinements](#architectural-refinements-orthogonal-to-the-conditioning-pattern) below.
- **[CaloTrilogy](https://arxiv.org/abs/2606.04165)** (Jiang, Qian, Pedro, Amram, Qu, Voetberg, 2026) — one/few-step flow-matching shower generation via an average-velocity integrator, a learned data-space generative prior (instead of starting from noise), and physics-guided loss terms on key observables. Orthogonal to the geometry question — a sampling-speed and training-objective contribution, not a geometry-representation one — but relevant if inference latency becomes a constraint.
- **[FM4CAL (MoE + PEFT foundation model)](https://arxiv.org/pdf/2603.28804)** — "Generalizable Foundation Models for Calorimetry via Mixtures-of-Experts and Parameter Efficient Fine Tuning." Trains on multiple detector geometries (ILC/CLIC) with **per-layer MoE routing** (which expert processes a token is decided per transformer layer, conditioned on the input) and adapts to new geometries via **LoRA**. A genuinely different pattern from CaloDiT-2/AllShowers' shared-weights-plus-embedding approach — see [Pattern 4](#geometry-conditioning-why-and-how) below.

## Graph-based models

- **[DeepTreeGAN](https://arxiv.org/abs/2311.12616)** → **[DeepTreeGANv2](https://arxiv.org/pdf/2312.00042)** — models shower point-cloud generation as a tree-structured growth process via graph convolutions (upscaling in generator, downscaling/pooling in critic). v2's iterative pooling targets the large dynamic range in energy that broke the original tree-GAN approach.
- **[CaloGraph](https://arxiv.org/abs/2402.11575)** (Kobylianskii et al., 2024) — graph diffusion model aimed specifically at *irregular-geometry* calorimeters (ATLAS CaloChallenge dataset). First graph-diffusion application in HEP; graph connectivity gives correct shower shapes without needing a fixed voxel grid.

## Adjacent point-cloud/graph literature from jet physics

Methodological ancestors of most of the above — worth having if you're evaluating architectures rather than just calorimeter results:

- **[MPGAN](https://arxiv.org/abs/2106.11535)** (Kansal et al., NeurIPS 2021) — message-passing GAN for particle clouds, introduced the JetNet benchmark; established that generic point cloud GANs don't work for HEP and message passing is needed.
- **[EPiC-GAN](https://arxiv.org/abs/2301.08128)** (Buhmann, Kasieczka, Thaler, 2023) — Deep-Sets-based, no pairwise message passing, much faster than graph/transformer approaches at large multiplicity. [Code](https://github.com/uhh-pd-ml/epic-gan).
- **[PC-JeDi](https://arxiv.org/abs/2303.05376)** and **[PC-Droid](https://arxiv.org/abs/2307.06836)** — score-based diffusion + transformer for particle-cloud jet generation; PC-Droid is 2-3 orders of magnitude faster than PC-JeDi/Delphes and trains on all jet types jointly — a similar "one model, many conditions" philosophy to AllShowers.
- **[EPiC-ly Fast (flow-matching)](https://arxiv.org/html/2310.00049)** and **[Pay Attention to Mean-Fields](https://arxiv.org/pdf/2408.04997)** — more recent flow-matching/attention variants in the same lineage, worth a skim for efficiency tricks.

## Datasets and surveys

- **[LEMURS dataset](https://arxiv.org/abs/2509.05108)** — 5M showers across 5 detector geometries (Par04SiW, Par04SciPb, ODD, FCCeeCLD, FCCeeALLEGRO), designed for multi-geometry/foundation-model work, HDF5 format similar in spirit to CaloChallenge. Companion paper: [A Generalisable Generative Model for Multi-Detector Calorimeter Simulation](https://arxiv.org/html/2509.07700).
- **[CaloChallenge 2022 community paper](https://arxiv.org/pdf/2410.21611)** — cross-comparison of essentially every method (including CaloPointFlow, DeepTreeGAN, CaloGraph) on the same datasets/metrics; best single reference for comparing point cloud/graph vs. voxel approaches head-to-head.
- **[A Comprehensive Evaluation of Generative Models in Calorimeter Shower Simulation](https://arxiv.org/html/2406.12898v1)** and **[Deep Generative Models for Detector Signature Simulation: A Taxonomic Review](https://arxiv.org/pdf/2312.09597)** — broader surveys.
- **[A First Full Physics Benchmark for Highly Granular Calorimeter Surrogates](https://arxiv.org/pdf/2511.17293)** — evaluates fast-sim surrogates (including point-cloud-based) not just on shower shapes but on downstream physics reconstruction.

# Geometry Conditioning: Why and How

Our model (`ddim-t_voxel/`) is a 3D U-Net with FiLM conditioning and a Transformer bottleneck, trained on a single fixed detector (CaloChallenge Dataset 2). Every design decision below is in service of one question: **what does the model need to be told about the detector so it can generate showers for a geometry it wasn't trained on?** That's a strictly harder ask than conditioning on energy/angle, because geometry changes the coordinate system itself, not just a scalar label. The case studies below split cleanly into two camps: models that treat geometry as *learned input* (generalizes, at a cost) and models that bake geometry into *fixed structure* (doesn't generalize, by construction).

## Case studies

**[CaloDiT-2](https://arxiv.org/html/2509.07700)** (`calodit2/`, arXiv:2509.07700) — built for exactly this:
- Encodes voxel positions in **physics-normalized units** (radiation lengths X₀, Molière radius Rₘ) instead of raw cell indices, so a voxel's "meaning" is comparable across detectors with totally different cell sizes/binning.
- Geometry enters as an explicit **one-hot vector G** (K known geometries + 1 reserved slot), projected and concatenated into the same conditioning embedding as (E, φ, θ), fed to every DiT block via adaptive LayerNorm — geometry is just another conditioning variable, not baked into the architecture.
- Generalization is achieved by **pretraining across 5 detector geometries (LEMURS)** then fine-tuning the reserved slot on a new geometry with 25× less data / 20× fewer steps. So it's transfer learning enabled by geometry conditioning, not pure zero-shot.

**[CaloGraph](https://arxiv.org/abs/2402.11575)** (arXiv:2402.11575) — a cautionary counter-example. Geometry isn't a conditioning input at all; it **is** the graph: nodes = fixed voxels with (η, φ, layer) features, edges = hand-built ring/layer neighbor connections. The graph is fixed per detector and the model is retrained per geometry — the paper's own conclusion says generalizing would require rethinking connectivity, and leaves that as future work. Lesson: baking geometry into hard structure (graph edges, grid shape) rather than into a conditioning vector loses cross-geometry transfer by construction.

**[CaloPointFlow II](https://arxiv.org/abs/2403.15782)** (arXiv:2403.15782) — VAE + two normalizing flows (LatentFlow + DeepSetFlow/PointFlow) with a novel CDF-Dequantization for discrete voxel energies, evaluated separately on CaloChallenge Dataset 2 and 3. No geometry-conditioning mechanism — trained per dataset. Its point-cloud representation is naturally more geometry-agnostic than a voxel grid (points are just coordinates+energy, no grid-shape assumption), but the paper doesn't exploit that for cross-geometry transfer — that combination is an open gap.

**[AllShowers](https://arxiv.org/pdf/2601.11716)** (arXiv:2601.11716) — geometry as global condition *plus* a structural attention constraint. Layer embeddings + Fourier position features encode where a voxel sits; custom attention masking enforces that energy deposition information only flows forward along the shower's physical propagation path (no attending from later layers back to earlier ones). Trained jointly across multiple detector geometries, which is real evidence that pattern 2 (below) scales past a single detector — but the paper's own results show generalization to a **genuinely unseen** geometry outside the training set still doesn't hold. Same ceiling as CaloDiT-2's fine-tuning slot, reached by a different architecture.

**[CaloArt](https://arxiv.org/abs/2605.12011)** (arXiv:2605.12011) — not a geometry-conditioning paper at all: one model per fixed detector, conditioned only on incident energy. Included here because its positional encoding is directly reusable: **3D axial RoPE**, where each patch token gets a `(z, r, α)` coordinate and rotary phases are computed per-axis, replacing a flat absolute positional embedding. See [architectural refinements](#architectural-refinements-orthogonal-to-the-conditioning-pattern) for how this composes with the patterns below.

**[FM4CAL](https://arxiv.org/pdf/2603.28804)** (arXiv:2603.28804) — a fourth pattern, distinct from "one shared network + a geometry embedding." Detector geometry determines **routing**: a per-layer Mixture-of-Experts decides which expert sub-network processes each token, conditioned on the input. New geometries are adapted with **LoRA** low-rank adapters instead of fine-tuning a reserved embedding slot. Trains on multiple ILC/CLIC geometries and evaluates adaptation to held-out ones.

## Four patterns for geometry conditioning

**1. Generate geometry-agnostic, then project onto geometry** — **CaloClouds**' approach. The model outputs a continuous 3D point cloud of energy deposits with no notion of cells at all; a fixed, *non-learned* binning step afterward projects those points onto whichever detector's voxel grid is wanted. Geometry never touches the network's weights — it's a deterministic renderer applied post-hoc.

**2. Geometry as a global condition** — **CaloDiT-2**'s and **AllShowers**' approach. Same weights for every detector, but a geometry embedding (one-hot / layer-and-position features) modulates every transformer block (adaLN or attention masking). Geometry is a learned input, but it's one vector or one structural rule describing the *whole* detector, shared across all tokens.

**3. Geometry as a local, per-cell condition** — **GAAM**'s approach (arXiv:2305.11531, confirmed from the paper text). At each autoregressive step, the model is fed the *individual cell's* size (Δη, Δφ) directly as a conditioning feature, alongside the previously-generated cells. Geometry participates cell-by-cell, not as one global code.

**4. Geometry as expert routing + lightweight adaptation** — **FM4CAL**'s approach. Geometry determines which sub-network (expert) handles a token, giving different geometries access to genuinely different computation rather than just a different modulation of the same computation. New geometries adapt via LoRA rather than fine-tuning an embedding.

**Tradeoffs:**
- **Pattern 1** gives generalization "for free" to any geometry since the projection step is unlearned — but it silently assumes different detectors only differ in *readout binning* of a shared physical field. It breaks if detectors differ in absorber material, sampling fraction, or thresholds — things that change the actual physics of deposition, not just how finely it's sampled.
- **Pattern 2** can learn genuine material/response differences (geometry is attended to, not just binned), and composes naturally with pretrain-then-finetune. But one global code per detector can't express how response varies *within* a single detector across depth/radius — the network has to implicitly infer that from the ID, unless (as in AllShowers) it's paired with a structural constraint like causal masking.
- **Pattern 3** is the most physically direct: the model can in principle learn a real scaling law of response vs. cell size/position. GAAM's own result is telling: it **interpolates** well between trained cell sizes but **fails to extrapolate** beyond them, meaning it's learning something closer to a memorized lookup than a closed-form physical law — a data-diversity problem, not a flaw in the pattern itself.
- **Pattern 4** gives more capacity than pattern 2 (different geometries can genuinely diverge in computation, not just modulation) and LoRA adaptation is cheaper than fine-tuning a reserved embedding slot. But it multiplies parameters (multiple experts), and the router still has to learn a geometry→expert assignment from data — it inherits the same "no true zero-shot for a wholly novel geometry" ceiling as pattern 2, just with a cheaper adaptation step once some data exists.

**Recommendation**: don't treat these as mutually exclusive — the strongest design combines 2 + 3, with 4 as a cheaper alternative to full fine-tuning once multiple geometries are available. Keep point-cloud/graph sparsity, but instead of a purely geometric post-hoc projection, feed **local geometry features per point** (cell size, layer material X₀, radial bin width at that location) the way GAAM does, *and* normalize coordinates into physical units plus pretrain across several geometries the way CaloDiT-2/AllShowers do. Pure projection (pattern 1) is the easiest to implement and "generalizes" trivially, but it's only correct if unseen geometries differ purely in binning, not in material stack — which is rarely true for real detector variants.

## Architectural refinements orthogonal to the conditioning pattern

CaloArt and AllShowers don't propose new *patterns* for the list above — CaloArt doesn't condition on geometry at all, and AllShowers is a pattern-2 instance — but both contribute mechanisms that change how well any pattern can use the geometry signal it's given:

- **Physical-unit RoPE (from CaloArt)**: CaloArt's 3D axial rotary position encoding computes phases from a `(z, r, α)` grid coordinate. As published, that's raw grid position, tied to one fixed detector. The portable version: compute those RoPE phases from **physically-normalized coordinates (X₀, Rₘ)** instead of raw indices. That makes the position encoding itself transfer across detectors with different binning, rather than relying entirely on a global geometry embedding to compensate for a position encoding that doesn't know the coordinates changed meaning.
- **Geometry-aware attention masking (from AllShowers)**: restricting attention to respect shower-depth causality is a free physical prior — it stops the network from learning spurious backward-in-depth correlations. Generalized beyond AllShowers' fixed layer-index masking, the mask can be computed from physical-unit distance/depth so it also transfers across geometries with a different number or thickness of layers, instead of being tied to a specific layer count.

Both are compatible with any of the four patterns above — they improve *how* the network uses positional/geometric information, independent of *what* geometry-identity signal (one-hot, continuous descriptor, expert routing) drives the global conditioning.

## GAAM details: dataset and backbone

**Dataset**: custom Geant4 simulation, ATLAS-like EM calorimeter geometry (not CaloChallenge). Three fixed longitudinal layers (inner: 5mm(η)×160mm(φ)×90mm depth; middle: 40mm×40mm×347mm; outer: 80mm×80mm×43mm). Single particle/energy: 10,000 simulated 65 GeV photons at the calorimeter center. "Different geometries" = different cell *segmentations* of these same fixed physical layers (e.g. inner layer from (48,4) up to (192,48) cells in η×φ) — i.e. re-binning one detector, not transferring across detectors with different materials/sampling fractions like LEMURS.

**Backbone**: one **MADE** (Masked Autoencoder for Distribution Estimation) autoregressive model per longitudinal layer. Inner layer: 1 masked FC layer + 1 1D conv layer. Middle/outer layers: 5 masked FC layers + 1 1D conv layer, GELU activations. Output: categorical distribution over N+1 discrete energy bins per cell, generated autoregressively over a spiral-ordered cell sequence, with each cell's (Δη, Δφ) size fed in as a conditioning feature at every step.

**Train/test split**: 12 training-geometry configurations across the three layers; evaluated on 5 held-out geometries — 2 **interpolation** cases ((24,24), (24,12)) within the training cell-size range, 3 **extrapolation** cases ((96,24), (6,6), (9,12)*) outside it. Confirms: interpolation works, extrapolation fails — the model is closer to a memorized lookup over trained cell sizes than a closed-form physical scaling law.

**Caveat for our project**: GAAM's "unseen geometry" is much narrower than CaloDiT-2's — re-binning a single fixed detector, not transferring across genuinely different detectors (materials, sampling fractions, absorber stacks). CaloDiT-2/LEMURS is the closer precedent if the goal is the latter; GAAM is more relevant if "unseen geometry" means finer/coarser voxelization of the same CaloChallenge Dataset 2 detector.

## Is one-hot geometry encoding (CaloDiT-2) too simplistic?

Yes — it's the weakest part of the design, not just aesthetically:
- OHE treats geometries as unrelated categorical labels with no notion of similarity: two nearly-identical detectors (e.g. same shape, different absorber) sit just as "far apart" in input space as two totally different ones. Any structure has to be learned by the shared network weights, not exposed by the encoding itself.
- The reserved K+1th slot for a genuinely new geometry carries **zero information** until fine-tuned — so CaloDiT-2 isn't doing zero-shot geometry generalization, it's doing fast few-shot adaptation from a good pretrained backbone. Real value, but a weaker claim than "conditioning on geometry properties." FM4CAL's LoRA-based adaptation (pattern 4) is a cheaper version of the same idea, but inherits the same limitation.

**Fix**: replace/augment the one-hot with a **continuous physical descriptor vector** (absorber X₀, Rₘ, sampling fraction, layer thickness, radial/angular bin widths, layer count). Turns geometry conditioning into regression over a physically meaningful space, so a genuinely unseen geometry gets a sensible zero-shot starting point via interpolation — the same way GAAM interpolates over cell size Δη/Δφ.

**Counter-argument**: real detector response isn't a clean smooth function of a handful of physical scalars — staggering, non-uniform sampling, edge effects add idiosyncrasies a low-dimensional continuous descriptor won't capture. One-hot + fine-tuning sidesteps needing a perfect physical parameterization, at the cost of requiring labeled data for every new geometry (no true zero-shot).

**Recommendation**: concatenate both — a continuous geometry-parameter vector *and* a learned per-geometry embedding (one-hot → embedding) in the same conditioning slot. The continuous part gives a reasonable prior/warm-start for a brand-new geometry pre-fine-tuning; the embedding absorbs whatever idiosyncratic response quirks the continuous descriptor can't express once fine-tuning data exists. Strictly more expressive than OHE alone, and degrades gracefully to CaloDiT-2's current behavior if the continuous part doesn't help.

## Zero-retraining geometry information checklist

If a model has already learned shower-development physics and only lacks geometry information, zero-shot transfer requires it to know both the **unit-conversion** (natural shower units → this detector's grid) and the **response/calibration** (true deposit → what gets read out):

**Longitudinal (depth) structure**
- Radiation length X₀ of the absorber (and active) material per layer — natural unit for EM longitudinal self-similarity
- Nuclear interaction length λ per layer — same role for hadronic showers
- Critical energy E꜀ of the absorber — sets shower-max depth via ln(E/E꜀)
- Physical thickness of each layer, in absolute units and in X₀/λ
- Number of layers and depth-ordering

**Transverse / angular structure**
- Molière radius Rₘ of the absorber — natural unit for lateral spread
- Cell size at each depth (Δr, Δφ or Δx, Δy) — binning granularity
- Detector topology: cylindrical/barrel vs endcap vs planar, projective vs non-projective towers — determines the coordinate system itself
- Inner/outer radial bounds (or lateral extent for planar) — edge/boundary effects

**Material & readout response**
- Sampling fraction per layer (active/total thickness ratio) — true deposit → visible signal
- Per-cell noise threshold / readout floor — what small deposits get zeroed
- Active vs passive material identity — affects local response even at fixed X₀/Rₘ (e.g. Si vs Sci)

**Global placement**
- Distance from interaction/entry point to the calorimeter's front face — angular divergence and shower-start position relative to the grid
- Angle of incidence / orientation convention — needed for a canonical "particle enters along z" frame

**Containment**
- Total depth in X₀ (and λ) — longitudinal leakage out the back
- Total lateral extent in Rₘ — lateral leakage out the sides

Missing unit-conversion params → shower shape right but misplaced/mis-scaled on the grid. Missing response params → shape and placement right but energy scale/sparsity pattern wrong.

## Actionable recipe for this project

Adapting the above to our 3D U-Net + FiLM + Transformer-bottleneck model (`ddim-t_voxel/`, currently single-geometry on CaloChallenge Dataset 2):

1. **Re-express voxel coordinates in physical units** (depth in X₀, radius in Rₘ) instead of raw indices, both for the FiLM conditioning input and for any positional encoding at the Transformer bottleneck.
2. **Upgrade the bottleneck's positional encoding to physical-unit RoPE** (adapting CaloArt): compute rotary phases from the X₀/Rₘ coordinates rather than raw grid position, so the encoding itself is portable across binnings, not just the FiLM condition.
3. **Add a continuous geometry-descriptor vector** to the FiLM conditioning MLP (X₀, Rₘ, sampling fraction, layer thickness, radial/angular bin counts), concatenated with — not replacing — a learned per-geometry embedding, per the one-hot critique above.
4. **Add geometry-aware attention masking** at the bottleneck (adapting AllShowers): restrict attention by physical depth/distance rather than raw layer index, so the causal prior also transfers across geometries with different layer counts.
5. **Once multiple geometries are available** (synthetic re-binnings of Dataset 2, or LEMURS), consider replacing full fine-tuning of a reserved slot with **LoRA-style adapters per new geometry** (FM4CAL) — cheaper adaptation, same underlying limitation (no true zero-shot to a wholly novel geometry).
6. **Add layer-wise optimal transport** as an auxiliary training loss (AllShowers) — orthogonal to the conditioning question, but catches layer-local shape mismatches a global metric misses.
7. **Pretrain across whatever geometries are available** before fine-tuning on a target unseen one — no conditioning trick substitutes for multi-geometry training data; single-geometry training won't generalize no matter how conditioning is set up.
