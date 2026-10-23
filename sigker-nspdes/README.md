# sigker-nspdes

Replica of [issaz/sigker-nsdes](https://github.com/issaz/sigker-nsdes) for Neural SPDEs.

## Idea

Train a Neural SPDE generator [2] without a discriminator by minimising a proper scoring rule [1]. A field u(t, x) is treated as a path in t; the score compares generated and real fields given the same initial condition u0.

- `sigker_nspdes.py`: data, signature kernel, generator, training.
- `sigker-nspdes-gl.ipynb`: 1D stochastic Ginzburg–Landau experiments.
- `sigker-nspdes-lob.ipynb`: limit order book (FI-2010) experiment ([1], Sec. 4.4 / App. B.5).

## Losses

Batch of b real fields y_g with u0_g = y_g(0, ·). For each g, generate M fields $x_{g,i} = \mathrm{Gen}_\theta(u0_g, \xi_{g,i})$. k is a signature kernel, $\varphi(u) = 0.1\,\mathrm{cumsum}_t(u)$.

**Conditional** ([1], eq. 8): minimum is the true law given u0.

$$
\mathcal{L} = \frac{1}{b}\sum_{g}\Big[\tfrac{1}{M(M-1)}\textstyle\sum_{i\neq j} k(\varphi(x_{g,i}),\varphi(x_{g,j})) - \tfrac{2}{M}\sum_i k(\varphi(x_{g,i}),\varphi(y_g))\Big]
$$

**Pooled** (reference, M = 1): every term mixes different u0s, so the minimum is the law pooled over u0.

$$
\mathcal{L}_{\text{pooled}} = \tfrac{1}{b(b-1)}\textstyle\sum_{g\neq h} k(\varphi(x_g),\varphi(x_h)) - \tfrac{2}{b^2}\sum_{g,h} k(\varphi(x_g),\varphi(y_h))
$$

**Energy** (β ∈ (0, 2), no φ, no PDE solves): conditional loss with k = −‖x − y‖^β.

$$
\mathcal{L}_{\text{energy}} = \frac{1}{b}\sum_{g}\Big[\tfrac{1}{M}\textstyle\sum_i \|x_{g,i}-y_g\|^\beta - \tfrac{1}{2M(M-1)}\sum_{i\neq j}\|x_{g,i}-x_{g,j}\|^\beta\Big]
$$

## Results

3000 steps, test u0s #0–2, 1000 generated vs 1000 resimulated fields each.

| | pooled | conditional (M=8) | energy (β=1, M=2) |
|---|---|---|---|
| Std at T, gen / true | 0.08–0.13 / 0.30 | 0.22–0.25 / 0.30 | 0.27–0.32 / 0.30 |
| \|u(0) − u0\| | 0.05–0.22 | 0.02–0.06 | 0.01–0.05 |
| KS accepted (p > 0.05) | 0% | 0–1% | 3–12% |
| Pooled loss on test (real-vs-real −2.73) | −2.83 | −2.49 | −2.17 |

- Pooled hits its optimum but collapses the per-u0 spread.
- Conditional recovers ~80% of the spread; energy ~100% and is fastest (0.07 s/step).
- KS still rejects most per-u0 marginals for all losses.
- Energy used 256 generated fields per step vs 128 for the others.

## Deliberate choices

- **Data:** du = (Δu + 3u − u³) dt + dW on [0, 1], Dirichlet; spectral Galerkin, semi-implicit Euler. T = 0.2, 32 × 64 grid. u0 = x(1−x) + 0.5 Σ_{j≤10} η_j sin(jπx)/j.
- **LOB data:** reference `lob_neurips.npy` (FI-2010, z-scored volumes × 0.5), windows of 32 events × 20 levels (bid L10…L1 | ask L1…L10), non-overlapping; last 10% held out as test. No true law given u0, so evaluation is population-level (KS on every (t, x) marginal, as [1] Fig. 3).
- **Multiscale σ:** RBF bandwidths median × {1/64, 1/8, 1}. The median alone (≈16) hides within-u0 noise (distances 0.03–24).
- **No batch norm:** the reference never calls `.eval()`, so BN couples samples.
- **Adam** instead of Adadelta (GBM warm-up: KS 95% vs 0%).
- **Custom signature kernel backward:** exact discrete adjoint; autograd needed ~50 GB.
- **Model:** hidden 16, Fourier modes 32 (x) / 16 (t), 4 Picard iterations, noise size 2, SWA from step 1500.

## References

1. Issa et al. *Non-adversarial training of Neural SDEs with signature kernel scores.* NeurIPS 2023. [arXiv:2305.16274](https://arxiv.org/abs/2305.16274)
2. Salvi et al. *Neural Stochastic PDEs.* NeurIPS 2022. [arXiv:2110.10249](https://arxiv.org/abs/2110.10249)
