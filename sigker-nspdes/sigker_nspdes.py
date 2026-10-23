"""From-scratch replica of the Neural SPDE experiment of github.com/issaz/sigker-nsdes
(notebooks/unconditional_nspde.ipynb), on synthetic 1D stochastic Ginzburg-Landau
data -- see sigker-nspdes.ipynb for data loading, training orchestration, and plots.

Method (Issa, Horvath, Lemercier & Salvi, "Non-adversarial training of Neural SDEs
with signature kernel scores", NeurIPS 2023, arXiv:2305.16274)
-------------------------------------------------------------------------------
A Neural SPDE generator (Salvi, Lemercier & Gerasimovics, arXiv:2110.10249)

    z = S_t * z0 + S * (F_theta(z) + G_theta(z) xi),     u = Q_theta(z),

solved by Picard iteration on the whole space-time domain, with the semigroup
S a learned kernel convolution parametrised in Fourier space, is trained *without
a discriminator* by minimising the expected signature-kernel score against real
fields u ~ Q:

    S(P_theta, u) = E_{X,X' ~ P_theta}[k_sig(X, X')] - 2 E_{X ~ P_theta}[k_sig(X, u)].

A field u(t, x) on a (dim_t, dim_x) grid is read as a *path in time* whose value at
each step is the whole spatial slice u(t, .) in R^{dim_x}. The static kernel is an
RBF on those slices (the reference's `rbf_id` kernel). As in the reference, the
score is computed on alpha * cumsum_t(u) rather than on u itself.

The signature kernel is the solution at (1, 1) of the Goursat PDE
d^2K/ds dt = <dx_s, dy_t>_kappa K, K(0, .) = K(., 0) = 1, solved here in pure torch
(SigKernelPDE) with the explicit 2nd-order scheme of the reference `sigkernel`
package, swept along anti-diagonals. Gradients come from a custom backward: the
exact discrete adjoint of the scheme.

Data: the stochastic Ginzburg-Landau (Phi^4_1) equation on [0, 1] with Dirichlet
boundary conditions,

    du = (Laplacian u + 3u - u^3) dt + sigma dW,

driven by space-time white noise, from random smooth initial conditions u0. The
generator receives the real u0 (reference: initial_point="given"). The reference's
score pools a batch over many u0s, which only targets the law pooled over u0s; with
cfg["m_per_u0"] = M the loss is instead the conditional score of arXiv:2305.16274
eq. 8 (M fields per u0, scored against that u0's real field), which targets the law
given u0.
"""

import math
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.optim.swa_utils import AveragedModel


# ---------------------------------------------------------------------------
# Data: stochastic Ginzburg-Landau, spectral Galerkin + semi-implicit Euler
# ---------------------------------------------------------------------------

def sine_basis(dim_x: int) -> torch.Tensor:
    """E[k, j] = sqrt(2) sin(j pi x_k) on the interior grid x_k = k / (dim_x + 1),
    j, k = 1..dim_x. Discrete orthogonality gives u = E u_hat, u_hat = E^T u / (dim_x + 1)."""
    k = torch.arange(1, dim_x + 1, dtype=torch.float64)
    x = k / (dim_x + 1)
    return math.sqrt(2) * torch.sin(math.pi * x[:, None] * k[None, :])


def random_initial_condition(n: int, dim_x: int, kappa: float = 0.5, n_modes: int = 10) -> torch.Tensor:
    """u0(x) = x(1 - x) + kappa * sum_{j <= n_modes} eta_j sin(j pi x) / j,  eta_j ~ N(0, 1).
    Smooth, random, and zero on the boundary (compatible with Dirichlet BCs)."""
    x   = torch.arange(1, dim_x + 1, dtype=torch.float64) / (dim_x + 1)
    j   = torch.arange(1, n_modes + 1, dtype=torch.float64)
    eta = torch.randn(n, n_modes, dtype=torch.float64)
    return x * (1 - x) + kappa * (eta / j) @ torch.sin(math.pi * j[:, None] * x[None, :])


def simulate_ginzburg_landau(u0: torch.Tensor, T: float = 0.2, dim_t: int = 32, sigma: float = 1.0,
                             substeps: int = 20, device="cpu") -> torch.Tensor:
    """Solve du = (u_xx + 3u - u^3) dt + sigma dW on [0, 1], u(t, 0) = u(t, 1) = 0.

    Spectral Galerkin in the Dirichlet sine basis (dim_x modes), semi-implicit Euler
    in time (Lord, Powell & Shardlow, "An Introduction to Computational Stochastic
    PDEs", Ch. 10): Laplacian implicit, reaction explicit,

        u_hat <- (u_hat + dt * f_hat(u) + sigma * dbeta) / (1 + dt (j pi)^2),

    with space-time white noise W = sum_j e_j beta_j truncated at dim_x modes.
    u0: (n, dim_x) on the interior grid. Returns (n, dim_t, dim_x), recorded every
    `substeps` fine steps on a uniform grid of [0, T]."""
    n, dim_x = u0.shape
    E     = sine_basis(dim_x).to(device)
    lam   = (math.pi * torch.arange(1, dim_x + 1, dtype=torch.float64, device=device)) ** 2
    dt    = T / ((dim_t - 1) * substeps)
    denom = 1.0 + dt * lam

    u   = u0.to(device, torch.float64)
    out = [u]
    for _ in range(dim_t - 1):
        for _ in range(substeps):
            u_hat = (u + dt * (3 * u - u ** 3)) @ E / (dim_x + 1)
            u_hat = u_hat + sigma * math.sqrt(dt) * torch.randn_like(u_hat)
            u     = (u_hat / denom) @ E.T
        out.append(u)
    return torch.stack(out, dim=1).float().cpu()


class GinzburgLandauDataset(torch.utils.data.Dataset):
    """Bank of stochastic Ginzburg-Landau solutions, stored as (N, dim_t, dim_x), with
    n_samples // n_per_ic random u0s and n_per_ic noise realisations of each (stored
    consecutively). No normalisation: field values are O(1)."""

    def __init__(self, n_samples=4096, dim_x=64, dim_t=32, T=0.2, sigma=1.0, kappa=0.5,
                 substeps=20, device="cpu", n_per_ic=1):
        self.dim_x, self.dim_t, self.T = dim_x, dim_t, T
        self.sigma, self.substeps      = sigma, substeps
        u0 = random_initial_condition(n_samples // n_per_ic, dim_x, kappa).repeat_interleave(n_per_ic, 0)
        self.fields = simulate_ginzburg_landau(u0, T, dim_t, sigma, substeps, device)
        self.x  = torch.arange(1, dim_x + 1, dtype=torch.float32) / (dim_x + 1)
        self.ts = torch.linspace(0, T, dim_t)

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        return self.fields[idx]

    def resimulate(self, u0: torch.Tensor, n: int, device="cpu") -> torch.Tensor:
        """n fresh solutions from one initial condition u0 (dim_x,): samples of the true
        conditional law given u0 -- only available because the data is synthetic."""
        return simulate_ginzburg_landau(u0[None].double().expand(n, -1), self.T, self.dim_t,
                                        self.sigma, self.substeps, device)


# FI-2010 (Ntakaris et al. 2018): 149 rows x n_events columns; row 4l + {0,1,2,3} is
# {ask price, ask volume, bid price, bid volume} at level l = 0..9. The space axis is
# price level, ordered bid L10..L1 | ask L1..L10, as in the reference lob_neurips.npy.
LOB_VOLUME_ROWS = list(range(39, 0, -4)) + list(range(1, 40, 4))


def read_fi2010_volumes(path: str) -> np.ndarray:
    """(n_events, 20) volumes from a FI-2010 .txt file; only the first 40 rows are parsed."""
    rows = []
    with open(path) as f:
        for _ in range(40):
            rows.append(np.array(f.readline().split(), dtype=np.float64))
    return np.stack(rows)[LOB_VOLUME_ROWS].T


class LOBDataset(torch.utils.data.Dataset):
    """FI-2010 limit order book volumes as fields (N, dim_t, dim_x=20), from non-overlapping
    windows of dim_t consecutive events.

    Reproduces the reference data/lob_neurips.npy (11325, 32, 20, 1) exactly from
    Train_Dst_NoAuction_ZScore_CF_9.txt with dim_t=32, scale=0.5 (checked: max |diff| = 0).
    The paper says T = 30, but the reference uses 32. `path` may also be the reference .npy.
    `split` = (start, stop) fractions of the windows, in chronological order (the paper uses
    all of them for training)."""

    def __init__(self, path: str, dim_t: int = 32, scale: float = 0.5, split=(0.0, 1.0)):
        if path.endswith(".npy"):
            fields = np.load(path)[..., 0]
        else:
            vol = read_fi2010_volumes(path)
            n   = vol.shape[0] // dim_t
            fields = scale * vol[:n * dim_t].reshape(n, dim_t, -1)
        n = fields.shape[0]
        fields = fields[int(split[0] * n):int(split[1] * n)]
        self.fields = torch.tensor(fields, dtype=torch.float32)
        self.dim_t, self.dim_x = self.fields.shape[1:]
        self.x  = torch.linspace(0, 1, self.dim_x)
        self.ts = torch.linspace(0, 1, self.dim_t)

    def __len__(self):
        return self.fields.shape[0]

    def __getitem__(self, idx):
        return self.fields[idx]


def infinite_loader(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Signature kernel (pure torch Goursat-PDE solver) and kernel score
# ---------------------------------------------------------------------------

def _antidiagonal(d: int, M: int, N: int, device):
    """Interior cells (i, j), 1 <= i <= M, 1 <= j <= N, on the anti-diagonal i + j = d."""
    i = torch.arange(max(1, d - N), min(M, d - 1) + 1, device=device)
    return i, d - i


class SigKernelPDE(torch.autograd.Function):
    """Solve the signature-kernel Goursat PDE for P path pairs at once.

    G : (P, m, n) static-kernel *increment* Gram, G[p, i, j] = <dx_i, dy_j>_kappa.
    The grid is refined by 2^dyadic_order in each direction (refined cell (i, j)
    reads base cell (i >> o, j >> o), scaled by 1 / 4^o), matching `sigkernel`.

    Explicit scheme on cell (i, j) with g = G[i-1, j-1]:
        K[i,j] = (K[i-1,j] + K[i,j-1]) A(g) - K[i-1,j-1] B(g),
        A(g) = 1 + g/2 + g^2/12,   B(g) = 1 - g^2/12,
    swept along anti-diagonals so all cells of a diagonal update in one op.
    Returns K[M, N], shape (P,).

    Backward is the *exact discrete adjoint* of that recursion (swept in reverse),
    so only the (M+1, N+1, P) grid K is kept -- plain autograd through the sweep
    would store ~6 tensors per diagonal, an order of magnitude more memory."""

    @staticmethod
    def forward(ctx, G, dyadic_order):
        P, m, n = G.shape
        f    = 2 ** dyadic_order
        M, N = m * f, n * f
        Gr   = G.permute(1, 2, 0) / (f * f)           # pairs last: each cell is a contiguous P-vector
        K    = G.new_ones(M + 1, N + 1, P)
        for d in range(2, M + N + 1):
            i, j = _antidiagonal(d, M, N, G.device)
            g    = Gr[(i - 1) // f, (j - 1) // f]
            g2   = g * g / 12.0
            K[i, j] = (K[i - 1, j] + K[i, j - 1]) * (1.0 + 0.5 * g + g2) - K[i - 1, j - 1] * (1.0 - g2)
        ctx.save_for_backward(G, K)
        ctx.dyadic_order = dyadic_order
        return K[M, N].clone()

    @staticmethod
    def backward(ctx, grad_out):
        G, K    = ctx.saved_tensors
        P, m, n = G.shape
        f    = 2 ** ctx.dyadic_order
        M, N = m * f, n * f
        Gr   = G.permute(1, 2, 0) / (f * f)
        Kbar = torch.zeros_like(K)
        Gbar = G.new_zeros(M, N, P)                  # gradient w.r.t. refined-grid g
        Kbar[M, N] = grad_out
        for d in range(M + N, 1, -1):
            i, j = _antidiagonal(d, M, N, G.device)
            kb   = Kbar[i, j]                        # final: all its dependents lie on diagonals > d
            g    = Gr[(i - 1) // f, (j - 1) // f]
            A    = 1.0 + 0.5 * g + g * g / 12.0
            B    = 1.0 - g * g / 12.0
            Gbar[i - 1, j - 1] = kb * ((K[i - 1, j] + K[i, j - 1]) * (0.5 + g / 6.0) + K[i - 1, j - 1] * (g / 6.0))
            # Separate statements: (i-1, j) of one cell is (i', j'-1) of its neighbour.
            Kbar[i - 1, j] += kb * A
            Kbar[i, j - 1] += kb * A
            Kbar[i - 1, j - 1] -= kb * B
        # Each base cell feeds f x f refined cells, each scaled by 1/f^2
        Gbar = Gbar.view(m, f, n, f, P).sum(dim=(1, 3)) / (f * f)
        return Gbar.permute(2, 0, 1), None


def solve_sigkernel_pde(G: torch.Tensor, dyadic_order: int = 0) -> torch.Tensor:
    return SigKernelPDE.apply(G, dyadic_order)


class SigKernel(nn.Module):
    """Signature kernel with a linear or RBF static kernel on path values in R^D.

    RBF: kappa(x, y) = exp(-|x - y|^2 / sigma) (sigkernel's convention: sigma divides
    the *squared* distance). For a field, D = dim_x and x, y are whole spatial slices,
    which is exactly sigkernel's `rbf_id` kernel. Increment Gram for RBF is the 2D
    finite difference kappa(x_{i+1}, y_{j+1}) + kappa(x_i, y_j) - kappa(x_i, y_{j+1}) - kappa(x_{i+1}, y_j).

    sigma may be a list: the signature kernel is then the *average* of the signature
    kernels at each bandwidth (the reference's SummedKernel), so differences at every
    scale in the list are resolved. One PDE solve per sigma.
    """

    def __init__(self, kernel_type: str = "rbf", sigma=1.0, dyadic_order: int = 1,
                 max_batch: int = 64):
        super().__init__()
        if kernel_type not in ("rbf", "linear"):
            raise ValueError(f"unknown kernel_type {kernel_type!r}")
        self.kernel_type  = kernel_type
        self.sigmas       = list(sigma) if isinstance(sigma, (list, tuple)) else [sigma]
        self.dyadic_order = dyadic_order
        self.max_batch    = max_batch

    def increment_grams(self, X: torch.Tensor, Y: torch.Tensor):
        """X (..., a, L1, D), Y (..., b, L2, D) -> one (..., a, b, L1-1, L2-1) increment Gram
        per sigma; leading dims (groups) are paired, not crossed."""
        if self.kernel_type == "linear":
            return [torch.einsum("...ipd,...jqd->...ijpq", X.diff(dim=-2), Y.diff(dim=-2))]
        sq = ((X ** 2).sum(-1)[..., :, None, :, None] + (Y ** 2).sum(-1)[..., None, :, None, :]
              - 2 * torch.einsum("...ipd,...jqd->...ijpq", X, Y)).clamp_min(0)
        grams = []
        for sigma in self.sigmas:
            k = torch.exp(-sq / sigma)
            grams.append(k[..., 1:, 1:] + k[..., :-1, :-1] - k[..., :-1, 1:] - k[..., 1:, :-1])
        return grams

    def _block(self, X, Y):
        shape = X.shape[:-2] + Y.shape[-3:-2]                              # (..., a, b)
        Ks = [solve_sigkernel_pde(G.flatten(0, -3), self.dyadic_order).view(shape)
              for G in self.increment_grams(X, Y)]
        return torch.stack(Ks).mean(0)

    def gram(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Gram matrix k_sig(X_i, Y_j), shape (a, b), computed in max_batch x max_batch blocks."""
        return torch.cat([
            torch.cat([self._block(Xc, Yc) for Yc in Y.split(self.max_batch)], dim=1)
            for Xc in X.split(self.max_batch)
        ], dim=0)

    def grouped_gram(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Within-group Gram k_sig(X[g, i], Y[g, j]), shape (G, a, b), for X (G, a, L, D) and
        Y (G, b, L, D); no cross-group pairs. Chunked over groups, ~max_batch^2 pairs each."""
        n = max(1, self.max_batch ** 2 // (X.shape[1] * Y.shape[1]))
        return torch.cat([self._block(Xc, Yc) for Xc, Yc in zip(X.split(n), Y.split(n))], dim=0)

    def forward(self, X, Y):
        return self.gram(X, Y)


def signature_kernel_score(kernel: SigKernel, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Expected signature kernel score E_{y ~ Y}[S(P_X, y)] estimated from generated
    paths X (m, L, D) and real paths Y (n, L, D):
        1/(m(m-1)) sum_{i!=j} k(x_i, x_j) - 2/(mn) sum_{i,j} k(x_i, y_j).
    Equals MMD^2(P_X, Y) minus the theta-independent E k(y, y') term, so it has the
    same gradients as the MMD but is not bounded below by 0."""
    m    = X.shape[0]
    K    = kernel.gram(X, torch.cat([X, Y], dim=0))   # one PDE sweep for both Grams
    K_XX, K_XY = K[:, :m], K[:, m:]
    return (K_XX.sum() - K_XX.diagonal().sum()) / (m * (m - 1)) - 2.0 * K_XY.mean()


def conditional_kernel_score(kernel: SigKernel, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Conditional score (arXiv:2305.16274, eq. 8): X (b, M, L, D) are M paths generated from
    the conditioning variable of each real path Y (b, L, D); each group is scored only
    against its own real path, and the per-group scores are averaged:
        1/b sum_g [ 1/(M(M-1)) sum_{i!=j} k(x_gi, x_gj) - 2/M sum_i k(x_gi, y_g) ].
    Proper for the law given the conditioning variable; needs M >= 2."""
    b, M = X.shape[:2]
    K    = kernel.grouped_gram(X, torch.cat([X, Y[:, None]], dim=1))   # (b, M, M+1), one sweep
    K_XX, K_XY = K[..., :M], K[..., M]
    off  = K_XX.sum((1, 2)) - K_XX.diagonal(dim1=1, dim2=2).sum(1)
    return (off / (M * (M - 1)) - 2.0 * K_XY.mean(1)).mean()


def energy_score(X: torch.Tensor, Y: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Energy score (calorimeter notes, eq. 63), with no transform Psi: X (b, M, L, D) are
    M fields generated from the conditioning variable of each real field Y (b, L, D), each
    read as one vector in R^{L D}, and the per-group estimates are averaged:
        1/b sum_g [ 1/M sum_i |x_gi - y_g|^beta - 1/(2M(M-1)) sum_{i!=j} |x_gi - x_gj|^beta ].
    Strictly proper for beta in (0, 2); unbiased for M >= 2."""
    M    = X.shape[1]
    X, Y = X.double().flatten(2), Y.double().flatten(1)
    d_XY = (X - Y[:, None]).norm(dim=-1).pow(beta)                    # (b, M)
    i, j = torch.triu_indices(M, M, 1, device=X.device)               # sum_{i!=j} = 2 sum_{i<j}
    d_XX = (X[:, i] - X[:, j]).norm(dim=-1).pow(beta)                 # (b, M(M-1)/2)
    return (d_XY.mean(1) - d_XX.sum(1) / (M * (M - 1))).mean()


def score_transform(fields: torch.Tensor, alpha) -> torch.Tensor:
    """Reference preprocessing before the kernel: alpha * cumsum over time (dim -2), in
    float64. alpha=None scores the raw fields."""
    fields = fields.double()
    return fields if alpha is None else alpha * torch.cumsum(fields, dim=-2)


def resolve_sigma(sigma, fields: torch.Tensor, alpha):
    """"median" -> median heuristic; "multiscale" -> median x (1/64, 1/8, 1): the
    within-u0 (noise-driven) slice distances are far below the population median and
    grow with t under the cumsum transform, so one bandwidth can't resolve them all.
    Floats and lists pass through."""
    if sigma == "median":
        return median_heuristic(fields, alpha)
    if sigma == "multiscale":
        med = median_heuristic(fields, alpha)
        return [med / 64, med / 8, med]
    return sigma


def median_heuristic(fields: torch.Tensor, alpha, n: int = 256) -> float:
    """Median squared distance between (transformed) spatial slices of different
    samples and times -- a natural scale for the RBF sigma."""
    Z = score_transform(fields[:n], alpha).flatten(0, 1)
    idx = torch.randperm(Z.shape[0])[:2048]
    return torch.cdist(Z[idx], Z[idx]).pow(2).median().item()


# ---------------------------------------------------------------------------
# Neural SPDE generator (fixed-point solver, as in the reference / 2110.10249)
# ---------------------------------------------------------------------------

class QWienerNoise(nn.Module):
    """Q-Wiener process W(t, x) = sum_j q_j^{1/2} e_j(x) beta_j(t) on the generator's grid,
    with e_j the Dirichlet sine basis (Lord, Powell & Shardlow, Example 10.31), as
    the reference's noise sampler. "white": q_j = 1 (space-time white noise);
    "colored": q_j = j^{-(2r + 1 + eps)} (spatially smooth). The generator is driven
    by the Wiener *path* (cumulative), not its increments, as in the reference."""

    def __init__(self, dim_x: int, noise_size: int, noise_type: str = "white", r: float = 2.0):
        super().__init__()
        self.noise_size = noise_size
        E = sine_basis(dim_x).float()
        if noise_type == "colored":
            j = torch.arange(1, dim_x + 1, dtype=torch.float32)
            E = E * j.pow(-(2 * r + 1 + 1e-3) / 2)
        elif noise_type != "white":
            raise ValueError(f"unknown noise_type {noise_type!r}")
        self.register_buffer("E", E)

    def sample(self, batch_size: int, ts: torch.Tensor) -> torch.Tensor:
        """Returns (batch, noise_size, dim_x, dim_t)."""
        dt = ts[1] - ts[0]
        dB = torch.randn(batch_size, self.noise_size, len(ts), self.E.shape[1], device=ts.device) * dt.sqrt()
        dB[:, :, 0] = 0.0
        B = torch.cumsum(dB, dim=2)                     # (batch, noise, t, modes)
        return (B @ self.E.T).transpose(-1, -2)         # (batch, noise, x, t)


class LocalOperators(nn.Module):
    """Pointwise (1x1 conv) F and G of the latent SPDE (d_t - L) z = F(z) + G(z) xi,
    as the reference's GeneratorFunc1d: [Conv 1x1 -> norm] x num_layers -> tanh.

    norm="batch" is the reference's BatchNorm2d. The reference never calls .eval(),
    so BN always normalises with *batch* statistics -- and since the same layers are
    reused at every Picard iteration, eval-mode running stats (averaged over all
    iterations) would not match any of them. With "batch" the generator is therefore
    always run in train mode (see NeuralSPDEGenerator.sampling_mode), which couples
    the samples of a batch. norm="none" drops it: samples are independent."""

    def __init__(self, noise_size: int, hidden_size: int, num_layers: int = 1, norm: str = "batch"):
        super().__init__()
        if norm not in ("batch", "none"):
            raise ValueError(f"unknown norm {norm!r}")
        self.noise_size  = noise_size
        self.hidden_size = hidden_size

        def stack(c_in, c_out):
            layers = []
            for k in range(num_layers):
                layers.append(nn.Conv2d(c_in if k == 0 else c_out, c_out, 1))
                if norm == "batch":
                    layers.append(nn.BatchNorm2d(c_out))
            return nn.Sequential(*layers, nn.Tanh())

        self.F = stack(hidden_size, hidden_size)
        self.G = stack(hidden_size, hidden_size * noise_size)

    def forward(self, z):
        """z: (batch, hidden, dim_x, dim_t) -> F (batch, hidden, x, t), G (batch, hidden, noise, x, t)."""
        b, h, nx, nt = z.shape
        return self.F(z), self.G(z).view(b, h, self.noise_size, nx, nt)


class KernelConvolution(nn.Module):
    """The semigroup S as a learned space-time convolution, parametrised by complex
    weights K_theta on the lowest (modes_x, modes_t) Fourier modes (fftshift-centred,
    as in the reference).
      forward(z)        : S * z  (convolution over space and time)
      forward_init(z0)  : S_t * z0 (K inverse-FFT'd along time, applied in space)."""

    def __init__(self, channels: int, modes_x: int, modes_t: int):
        super().__init__()
        self.modes   = (modes_x, modes_t)
        self.weights = nn.Parameter(torch.rand(channels, channels, modes_x, modes_t, dtype=torch.cfloat) / channels ** 2)

    def _window(self, n, m):
        return slice(n // 2 - m // 2, n // 2 + m // 2)

    def forward(self, z):
        b, c, nx, nt = z.shape
        sx, st = self._window(nx, self.modes[0]), self._window(nt, self.modes[1])
        z_ft   = torch.fft.fftshift(torch.fft.fftn(z, dim=(2, 3)), dim=(2, 3))
        out_ft = torch.zeros_like(z_ft)
        out_ft[:, :, sx, st] = torch.einsum("bixt,ijxt->bjxt", z_ft[:, :, sx, st], self.weights)
        return torch.fft.ifftn(torch.fft.ifftshift(out_ft, dim=(2, 3)), dim=(2, 3)).real

    def forward_init(self, z0_path):
        b, c, nx, nt = z0_path.shape
        sx      = self._window(nx, self.modes[0])
        weights = torch.fft.ifftn(torch.fft.ifftshift(self.weights, dim=[-1]), dim=[-1], s=nt)   # K_t
        z_ft    = torch.fft.fftshift(torch.fft.fftn(z0_path, dim=[2]), dim=[2])
        out_ft  = torch.zeros_like(z_ft)
        out_ft[:, :, sx] = torch.einsum("bixt,ijxt->bjxt", z_ft[:, :, sx], weights)
        return torch.fft.ifftn(torch.fft.ifftshift(out_ft, dim=[2]), dim=[2]).real


class NeuralSPDEGenerator(nn.Module):
    """u0 -> lift z0 = A u0 -> Picard iterations of the mild solution
        z_{k+1} = S_t * z0 + S * (F(z_k) + G(z_k) xi)
    -> readout Q(z) (Linear-ReLU-Linear). Returns fields (batch, dim_t, dim_x)."""

    def __init__(self, dim_x: int, noise_size=2, hidden_size=16, num_layers=1, modes_x=16, modes_t=16,
                 n_iter=4, noise_type="white", readout_size=128, norm="batch"):
        super().__init__()
        self.n_iter  = n_iter
        self.norm    = norm
        self.noise   = QWienerNoise(dim_x, noise_size, noise_type)
        self.lift    = nn.Linear(1, hidden_size)
        self.func    = LocalOperators(noise_size, hidden_size, num_layers, norm)
        self.conv    = KernelConvolution(hidden_size, modes_x, modes_t)
        self.readout = nn.Sequential(nn.Linear(hidden_size, readout_size), nn.ReLU(), nn.Linear(readout_size, 1))

    def sampling_mode(self):
        """Mode for evaluation / sampling: eval() normally, but train() with BatchNorm,
        which the reference always runs on batch statistics (see LocalOperators)."""
        return self.train(self.norm == "batch")

    def forward(self, u0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """u0: (batch, dim_x), ts: (dim_t,) -> (batch, dim_t, dim_x)."""
        b, nt = u0.shape[0], len(ts)
        xi = self.noise.sample(b, ts)                                     # (b, noise, x, t)
        z0 = self.lift(u0.unsqueeze(-1)).permute(0, 2, 1)                 # (b, hidden, x)
        z0_path = self.conv.forward_init(z0.unsqueeze(-1).expand(-1, -1, -1, nt))
        z = z0_path
        for _ in range(self.n_iter):
            F_z, G_z = self.func(z)
            z = z0_path + self.conv(F_z + torch.einsum("bhnxt,bnxt->bhxt", G_z, xi))
        return self.readout(z.permute(0, 3, 2, 1)).squeeze(-1)           # (b, t, x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_generator(cfg: dict) -> NeuralSPDEGenerator:
    return NeuralSPDEGenerator(
        dim_x        = cfg["dim_x"],
        noise_size   = cfg["noise_size"],
        hidden_size  = cfg["hidden_size"],
        num_layers   = cfg["num_layers"],
        modes_x      = cfg["modes_x"],
        modes_t      = cfg["modes_t"],
        n_iter       = cfg["n_iter"],
        noise_type   = cfg["noise_type"],
        readout_size = cfg["readout_size"],
        norm         = cfg["norm"],
    )


def build_kernel(cfg: dict) -> SigKernel:
    return SigKernel(cfg["kernel_type"], cfg["sigma"], cfg["dyadic_order"], cfg["max_batch"])


def score_loss(generator, kernel, ts, real, cfg):
    """Kernel score of fields generated from the real batch's own u0s against that batch.
    Without cfg["m_per_u0"]: one field per u0, scored against the whole batch (the
    reference's pooled score, proper only for the law pooled over u0s). With
    cfg["m_per_u0"] = M >= 2: M fields per u0, each group scored only against its own
    real field (conditional score, proper for the law given u0). cfg["score"] = "energy"
    uses the energy score of the calorimeter notes (eq. 63) on the raw fields instead of
    the signature kernel score; it needs cfg["m_per_u0"] >= 2 and reads cfg["beta"]."""
    M = cfg.get("m_per_u0")
    if cfg.get("score", "sigkernel") == "energy":
        if not M or M < 2:
            raise ValueError('cfg["score"] = "energy" needs cfg["m_per_u0"] >= 2')
        fake = generator(real[:, 0].repeat_interleave(M, 0), ts).view(real.shape[0], M, *real.shape[1:])
        return energy_score(fake, real, cfg.get("beta", 1.0))
    if not M:
        fake = generator(real[:, 0], ts)
        return signature_kernel_score(kernel, score_transform(fake, cfg["alpha"]),
                                      score_transform(real, cfg["alpha"]))
    fake = generator(real[:, 0].repeat_interleave(M, 0), ts).view(real.shape[0], M, *real.shape[1:])
    return conditional_kernel_score(kernel, score_transform(fake, cfg["alpha"]),
                                    score_transform(real, cfg["alpha"]))


def train(cfg: dict, train_loader, test_loader=None):
    """Step-based training (as the reference). cfg["mode"] must be "train" (fresh) or
    "resume" (from the latest checkpoint in cfg["ckpt_dir"]). Returns (generator,
    averaged_generator, loss_history, test_loss_history), where the averaged
    generator is the stochastic weight average from cfg["swa_step_start"] on."""
    if cfg["mode"] not in ("train", "resume"):
        raise ValueError(f'train() requires cfg["mode"] in ("train", "resume"), got {cfg["mode"]!r}')

    device   = cfg["device"]
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 0))

    generator = build_generator(cfg).to(device)
    averaged  = AveragedModel(generator)   # BN running stats are never used (see LocalOperators)
    kernel    = build_kernel(cfg)
    ts        = train_loader.dataset.ts.to(device)

    print(f"  Parameters : {sum(p.numel() for p in generator.parameters()) / 1e3:.1f} K")

    opt = getattr(torch.optim, cfg["optimizer"])(generator.parameters(), lr=cfg["lr"],
                                                 weight_decay=cfg["weight_decay"])

    start_step = 0
    loss_history, test_loss_history = [], []

    if cfg["mode"] == "resume":
        latest = sorted(ckpt_dir.glob("ckpt*.pt"))
        if not latest:
            raise FileNotFoundError(f'mode="resume" but no checkpoints found in {ckpt_dir}')
        state = torch.load(latest[-1], map_location=device)
        generator.load_state_dict(state["generator"])
        averaged.load_state_dict(state["averaged"])
        opt.load_state_dict(state["opt"])
        start_step        = state["step"] + 1
        loss_history      = state.get("loss_history", [])
        test_loss_history = state.get("test_loss_history", [])
        print(f"Resumed from {latest[-1]}  (step {start_step})")

    batches = infinite_loader(train_loader)
    for step in range(start_step, cfg["steps"]):
        generator.train()
        real = next(batches).to(device)
        loss = score_loss(generator, kernel, ts, real, cfg)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), cfg.get("grad_clip", float("inf")))
        if torch.isfinite(grad_norm):
            opt.step()
        else:
            print(f"  [skip] non-finite grad norm at step {step + 1}")
        opt.zero_grad()

        if step >= cfg["swa_step_start"]:
            averaged.update_parameters(generator)

        loss_history.append(loss.item())
        if wandb.run is not None:
            wandb.log({"train_loss": loss.item(), "grad_norm": grad_norm.item()}, step=step + 1)

        if (step + 1) % cfg["log_every"] == 0:
            window = loss_history[-cfg["log_every"]:]
            msg = f"Step {step + 1:5d}/{cfg['steps']}  |  loss={np.mean(window):.5f}  grad_norm={grad_norm.item():.3f}"
            if test_loader is not None:
                eval_gen = averaged.module if step >= cfg["swa_step_start"] else generator
                test_loss = evaluate(eval_gen, kernel, ts, test_loader, cfg)
                test_loss_history.append((step + 1, test_loss))
                msg += f"  test_loss={test_loss:.5f}"
                if wandb.run is not None:
                    wandb.log({"test_loss": test_loss}, step=step + 1)
            print(msg)

        if (step + 1) % cfg["save_every"] == 0 or step + 1 == cfg["steps"]:
            if not math.isfinite(loss.item()):
                print(f"  -> step {step + 1}: non-finite loss, skipping checkpoint save")
                continue
            ckpt_path = ckpt_dir / f"ckpt{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_step{step + 1:06d}.pt"
            torch.save({
                "step":              step,
                "generator":         generator.state_dict(),
                "averaged":          averaged.state_dict(),
                "opt":               opt.state_dict(),
                "loss_history":      loss_history,
                "test_loss_history": test_loss_history,
                "cfg":               cfg,
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}")

    return generator, averaged.module, loss_history, test_loss_history


@torch.no_grad()
def evaluate(generator, kernel, ts, loader, cfg, max_batches: int = 4) -> float:
    """Mean kernel score over (up to) `max_batches` test batches."""
    generator.sampling_mode()
    losses = []
    for b, real in enumerate(loader):
        if b >= max_batches:
            break
        losses.append(score_loss(generator, kernel, ts, real.to(cfg["device"]), cfg).item())
    return float(np.mean(losses))


def load_checkpoint(cfg: dict, averaged: bool = True):
    """Load the latest checkpoint in cfg["ckpt_dir"]; returns
    (generator, loaded_cfg, loss_history, test_loss_history).
    With averaged=True returns the SWA-averaged weights when SWA has started."""
    device = cfg["device"]
    ckpts  = sorted(Path(cfg["ckpt_dir"]).glob("ckpt*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {cfg['ckpt_dir']}")
    state      = torch.load(ckpts[-1], map_location=device)
    loaded_cfg = state["cfg"]
    print(f"Loading {ckpts[-1]}  (step {state['step'] + 1})")

    generator = build_generator(loaded_cfg).to(device)
    if averaged and state["step"] >= loaded_cfg["swa_step_start"]:
        swa = AveragedModel(generator)
        swa.load_state_dict(state["averaged"])
        generator = swa.module
    else:
        generator.load_state_dict(state["generator"])
    generator.sampling_mode()
    return generator, loaded_cfg, state.get("loss_history", []), state.get("test_loss_history", [])


@torch.no_grad()
def generate(generator, u0: torch.Tensor, ts: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Fields (n, dim_t, dim_x) from initial conditions u0 (n, dim_x), on the CPU.
    With norm="batch", outputs depend on the batch's composition (batch statistics)."""
    generator.sampling_mode()
    device = ts.device
    return torch.cat([generator(u.to(device), ts).cpu() for u in u0.split(batch_size)], dim=0)


def ks_marginals(fake: torch.Tensor, real: torch.Tensor, alpha: float = 0.05):
    """Two-sample KS test on every (t, x) marginal for t >= 1 (t = 0 is the given u0).
    fake, real: (n, dim_t, dim_x). Returns (p-values (dim_t-1, dim_x), fraction accepted)."""
    from scipy.stats import ks_2samp
    pvals = ks_2samp(fake[:, 1:].numpy(), real[:, 1:].numpy(), axis=0).pvalue
    return pvals, float((pvals > alpha).mean())
