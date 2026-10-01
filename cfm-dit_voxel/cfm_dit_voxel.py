"""Model, flow-matching, and training/generation code for the CFM+DiT calorimeter
shower model (jointly trained on CaloChallenge Dataset 2 SiW and LEMURS Par04SiW,
zero-shot evaluated on held-out LEMURS FCCeeALLEGRO). See cfm-dit_voxel.ipynb for
data loading, training/eval orchestration, and plotting.
"""

import math
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding (Vaswani et al.); works for continuous t as well as integer steps."""
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class TimeEmbedding(nn.Module):
    """Embeds the continuous flow-matching time t (scaled by 1000 for frequency spread)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        return self.net(sinusoidal_embedding(t, self.dim))  # (B, dim*4)


class AngleEmbedding(nn.Module):
    """Embeds incident (phi, theta) via sin/cos features -> MLP (avoids the phi=0/2pi
    discontinuity a raw-radian input would have). Used identically for both geometries;
    CaloChallenge Dataset 2 always supplies the fixed (phi=0, theta=pi/2)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, phi: torch.Tensor, theta: torch.Tensor):
        feats = torch.stack([phi.sin(), phi.cos(), theta.sin(), theta.cos()], dim=-1)  # (B, 4)
        return self.net(feats)


# ── Patchify / unpatchify ──────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """Non-overlapping 3D patch embedding (large patches, CaloArt-style)."""

    def __init__(self, in_channels: int, patch_size: tuple, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)                       # (B, embed_dim, D_p, H_p, W_p)
        grid_shape = x.shape[2:]
        x = einops.rearrange(x, 'b c d h w -> b (d h w) c')
        return x, grid_shape


class FinalLayer3D(nn.Module):
    """adaLN-Zero output head: projects tokens back to patches and unpatchifies to the grid."""

    def __init__(self, embed_dim: int, patch_size: tuple, out_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 2))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
        patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        self.out_proj = nn.Linear(embed_dim, patch_vol * out_channels)
        nn.init.zeros_(self.out_proj.weight)   # zero-init -> v_theta(x,0)=0 at init
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, grid_shape: tuple):
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.out_proj(x)                   # (B, N, patch_vol*out_channels)
        d_p, h_p, w_p = grid_shape
        pd, ph, pw = self.patch_size
        x = einops.rearrange(
            x, 'b (d h w) (pd ph pw c) -> b c (d pd) (h ph) (w pw)',
            d=d_p, h=h_p, w=w_p, pd=pd, ph=ph, pw=pw, c=self.out_channels,
        )
        return x


# ── 3D axial RoPE (CaloArt-inspired) ───────────────────────────────────────────

class AxialRoPE3D(nn.Module):
    """
    3D axial rotary position embedding.

    Splits each attention head's dimension into 3 equal chunks — one per patch-grid
    axis (depth/layer, angle, radius) — and applies standard (NeoX-style) RoPE
    rotation to each chunk using that axis's *patch-grid* coordinate as the rotary
    position. This is the raw-grid-position version described for CaloArt. Even
    though this model already trains jointly on two geometries (SiW + Par04SiW),
    both share the same (45,16,9) voxel grid, so raw patch-grid position is still a
    consistent coordinate system across them. Making this transfer across detectors
    with genuinely different binning/grid shapes would mean computing the per-axis
    position from physical units (X0, Rm) instead of the patch index — not yet
    needed here, but a drop-in swap to this one class if a third, differently-binned
    geometry is ever added (see top-level README.md, "Architectural refinements"
    section).
    """

    def __init__(self, head_dim: int, grid_shape: tuple, base: float = 10000.0):
        super().__init__()
        assert head_dim % 6 == 0, "head_dim must be divisible by 6 for 3D axial RoPE (2 * 3 axes)"
        self.axis_dim = head_dim // 3
        self.grid_shape = grid_shape

        inv_freq = 1.0 / (base ** (torch.arange(0, self.axis_dim, 2).float() / self.axis_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        d_p, h_p, w_p = grid_shape
        coords = torch.stack(torch.meshgrid(
            torch.arange(d_p), torch.arange(h_p), torch.arange(w_p), indexing="ij"
        ), dim=-1).reshape(-1, 3).float()      # (N, 3) — per-token (d, h, w) patch coord
        self.register_buffer("coords", coords, persistent=False)

    def _freqs_for_axis(self, pos):
        freqs = pos[:, None] * self.inv_freq[None, :]   # (N, axis_dim//2)
        return torch.cat([freqs, freqs], dim=-1)         # (N, axis_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (B, heads, N, head_dim)
        def apply(x):
            chunks = x.split(self.axis_dim, dim=-1)     # one chunk per axis
            out = []
            for axis, xc in enumerate(chunks):
                pos = self.coords[:, axis].to(x.device)
                freqs = self._freqs_for_axis(pos)
                cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
                x1, x2 = xc.chunk(2, dim=-1)
                rot = torch.cat([-x2, x1], dim=-1)
                out.append(xc * cos + rot * sin)
            return torch.cat(out, dim=-1)
        return apply(q), apply(k)


class RMSNorm(nn.Module):
    """QK-norm (Dehghani et al. 2023, ViT-22B; Gemma 2): RMS-normalizes each head's
    q/k vector before the dot product, so attention logits stay bounded regardless of
    how large activation scale grows elsewhere in training. Not affected by RoPE's
    placement: RoPE is a per-axis rotation and preserves the head_dim L2 norm, so
    normalizing before vs. after rope() is mathematically equivalent -- applied before
    here to match the usual placement."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class Attention3DRoPE(nn.Module):
    """Multi-head self-attention over patch tokens with 3D axial RoPE on Q/K.

    QK-norm (RMS-normalizing q/k per head before the dot product) keeps attention
    logits bounded regardless of upstream activation scale, preventing the runaway
    gradient blow-up that plain dot-product attention is prone to during extended
    training. `grad_skip_mult` in `train()` remains as a safety net against a future
    recurrence, but QK-norm is what addresses the root cause rather than just
    rejecting the resulting bad steps.
    """

    def __init__(self, embed_dim: int, num_heads: int, grid_shape: tuple, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.rope = AxialRoPE3D(self.head_dim, grid_shape)
        self.dropout = dropout

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                # (B, heads, N, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ── DiT block (adaLN-Zero conditioning, Peebles & Xie 2023) ────────────────────

class DiTBlock3D(nn.Module):
    """Pre-LN transformer block with adaLN-Zero modulation from the (t, E, geometry) embedding."""

    def __init__(self, embed_dim: int, num_heads: int, grid_shape: tuple,
                 mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn  = Attention3DRoPE(embed_dim, num_heads, grid_shape, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(approximate="tanh"),
            nn.Dropout(dropout), nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )
        # 6 = (shift, scale, gate) x (attn, mlp)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, embed_dim * 6))
        nn.init.zeros_(self.adaLN[-1].weight)   # zero-init -> block is identity at init
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.attn(h)
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


# ── Full backbone ───────────────────────────────────────────────────────────────

class ConditionalFlowMatchingDiT(nn.Module):
    """
    CaloArt-inspired Diffusion Transformer, trained as a conditional flow-matching
    velocity predictor. Pure transformer — no convolutional U-Net path.

    Conditioning: adaLN-Zero on every block from a merged embedding of
      - flow-matching time t (continuous, in [0, 1])
      - incident energy log10(E_inc / GeV)
      - incident angle (phi, theta), via sin/cos features — real per-sample values for
        LEMURS Par04SiW, a fixed (phi=0, theta=pi/2) constant for CaloChallenge
        Dataset 2 (perpendicular incidence, azimuthally symmetric)
      - a geometry embedding (n_geometries=3: 0=SiW/CaloChallenge Dataset 2,
        1=Par04SiW, 2=LEMURS FCCeeALLEGRO — held out, never trained, used only
        for a zero-shot generalisation eval), matching CaloDiT-2's
        one-hot-embedding conditioning pattern and held-out-detector methodology

    Args
        x_t     : (B, 1, D, H, W)  interpolated shower at flow-time t
        t       : (B,)              continuous flow time in [0, 1]
        cond    : (B,)              log10(E_inc / GeV)
        phi, theta : (B,)           incident angle in radians
        geom_id : (B,) long or None — geometry index; defaults to all-zeros

    Returns
        v_pred : (B, 1, D, H, W)  predicted velocity field x1 - x0
    """

    def __init__(self, voxel_shape, patch_size=(5, 4, 3), embed_dim=384, depth=8,
                 num_heads=8, mlp_ratio=4, cond_dim=128, n_geometries=3, dropout=0.0):
        super().__init__()
        self.voxel_shape = voxel_shape
        self.patch_embed = PatchEmbed3D(1, patch_size, embed_dim)
        grid_shape = tuple(v // p for v, p in zip(voxel_shape, patch_size))

        self.time_emb    = TimeEmbedding(embed_dim // 4)     # dim*4 == embed_dim
        self.energy_proj = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, embed_dim)
        )
        self.angle_emb   = AngleEmbedding(embed_dim)
        self.geom_embed  = nn.Embedding(n_geometries, embed_dim)
        self.cond_merge  = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        self.blocks = nn.ModuleList([
            DiTBlock3D(embed_dim, num_heads, grid_shape, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.final = FinalLayer3D(embed_dim, patch_size, out_channels=1)

    def forward(self, x_t, t, cond, phi, theta, geom_id=None):
        if geom_id is None:
            geom_id = torch.zeros(x_t.shape[0], dtype=torch.long, device=x_t.device)

        tokens, grid_shape = self.patch_embed(x_t)

        t_emb = self.time_emb(t * 1000.0)                # (B, embed_dim)
        e_emb = self.energy_proj(cond.unsqueeze(-1))      # (B, embed_dim)
        a_emb = self.angle_emb(phi, theta)                # (B, embed_dim)
        g_emb = self.geom_embed(geom_id)                  # (B, embed_dim)
        c     = self.cond_merge(torch.cat([t_emb, e_emb, a_emb, g_emb], dim=-1))  # (B, embed_dim)

        for block in self.blocks:
            tokens = block(tokens, c)

        return self.final(tokens, c, grid_shape)


class RectifiedFlow:
    """
    Conditional Flow Matching via the rectified-flow linear path
    (Liu et al., 2022, arXiv:2209.03003; Lipman et al., 2023 CFM with sigma_min=0).

    Training: sample t~U(0,1) and x0~N(0,I), interpolate x_t = (1-t)*x0 + t*x1,
              regress the (t-independent along the path) target velocity u_t = x1 - x0.
    Sampling: integrate the ODE dx/dt = v_theta(x, t, cond, phi, theta, geom_id) from
              t=0 to t=1, via Euler (1 NFE/step) or Heun/RK2 (2 NFE/step, better
              quality at low steps). x is clamped to [-1,1] after every step, not just
              the final one: an uncorrected mid-integration excursion outside [-1,1]
              passes through the log-normalisation's exp() inverse and turns into a
              huge fake MeV value at generation time, so clamping throughout the
              trajectory (not just at the end) avoids that failure mode entirely.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def training_loss(self, model, x1, cond, phi, theta, geom_id=None):
        """Sample random t, interpolate x0->x1, return MSE(v_pred, x1 - x0)."""
        B    = x1.shape[0]
        t    = torch.rand(B, device=x1.device)
        x0   = torch.randn_like(x1)
        t_   = t[:, None, None, None, None]
        x_t  = (1 - t_) * x0 + t_ * x1
        target = x1 - x0
        v_pred = model(x_t, t, cond, phi, theta, geom_id)
        return F.mse_loss(v_pred, target)

    @torch.no_grad()
    def sample(self, model, shape, cond, phi, theta, geom_id=None, steps: int = 50, method: str = "heun"):
        """Integrate dx/dt = v_theta from t=0 (noise) to t=1 (data)."""
        device = self.device
        x  = torch.randn(shape, device=device)
        ts = torch.linspace(0.0, 1.0, steps + 1, device=device)

        for i in range(steps):
            t_cur, t_next = ts[i], ts[i + 1]
            dt    = t_next - t_cur
            t_vec = t_cur.expand(shape[0])
            v1    = model(x, t_vec, cond, phi, theta, geom_id)

            if method == "euler":
                x = x + dt * v1
            elif method == "heun":
                x_euler     = x + dt * v1
                t_next_vec  = t_next.expand(shape[0])
                v2          = model(x_euler, t_next_vec, cond, phi, theta, geom_id)
                x           = x + dt * 0.5 * (v1 + v2)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Clamp every step, not just the final sample -- see class docstring.
            x = x.clamp(-1.0, 1.0)

        return x


def train(cfg: dict, train_loader, test_loader=None):
    device    = cfg["device"]
    ckpt_dir  = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Fix model init + per-epoch DataLoader shuffle order so a retry is comparable
    # instead of depending on luck.
    torch.manual_seed(cfg.get("seed", 0))

    model = ConditionalFlowMatchingDiT(
        voxel_shape  = cfg["voxel_shape"],
        patch_size   = cfg["patch_size"],
        embed_dim    = cfg["embed_dim"],
        depth        = cfg["depth"],
        num_heads    = cfg["num_heads"],
        mlp_ratio    = cfg["mlp_ratio"],
        cond_dim     = cfg["cond_dim"],
        n_geometries = cfg["n_geometries"],
        dropout      = cfg["dropout"],
    ).to(device)

    if cfg.get("compile", False):
        model = torch.compile(model)

    flow = RectifiedFlow(device=device)

    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    vram_gib = (param_bytes + buffer_bytes) / 1024**3
    print(f"  Parameters : {n_params / 1e6:.2f} M")
    print(f"  VRAM       : {vram_gib:.3f} GiB")

    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    # Linear warmup (helps avoid an attention-logit collapse at full peak LR from
    # step 1) followed by cosine decay over the remaining epochs.
    warmup_epochs = cfg.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup = LinearLR(opt, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(opt, T_max=cfg["epochs"] - warmup_epochs, eta_min=1e-6)
        sched  = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        sched = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-6)

    # bfloat16 (not the torch.autocast default of float16): same exponent range as
    # fp32, so it doesn't overflow to inf the way float16 does on large attention
    # logits/activations. GradScaler is kept for the finite-grad-norm guard below,
    # but bf16 itself doesn't need loss scaling.
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    start_epoch = 0
    loss_history = []
    test_loss_history = []

    if not cfg['scratch']:
        latest = sorted(ckpt_dir.glob("ckpt*.pt"))
        if latest:
            state = torch.load(latest[-1], map_location=device)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            print(f"Resumed from {latest[-1]}  (epoch {start_epoch})")

    grad_skip_mult = cfg.get("grad_skip_mult", float("inf"))

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_skipped = 0        # non-finite grad norm (inf/nan)
        n_skipped_large = 0  # finite but > grad_clip * grad_skip_mult
        grad_norm_sum = 0.0
        grad_norm_max = 0.0

        for x1, cond, phi, theta, geom_id in train_loader:
            x1, cond   = x1.to(device), cond.to(device)
            phi, theta = phi.to(device), theta.to(device)
            geom_id    = geom_id.to(device)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                loss = flow.training_loss(model, x1, cond, phi, theta, geom_id)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            grad_norm_finite = torch.isfinite(grad_norm)
            # clip_grad_norm_'s own norm computation can overflow to inf/nan even when
            # every individual gradient was finite -- that happens *after* GradScaler's
            # unscale_() already checked for inf/nan, so scaler.step() alone won't catch
            # it and will silently bake NaNs into the weights. Guard explicitly.
            #
            # isfinite() alone isn't enough though: a pre-clip norm in the millions is
            # still "finite", still gets clipped-then-applied, and clip_grad_norm_'s
            # rescale is itself a huge-magnitude operation in bf16 that can corrupt the
            # update and cascade into every subsequent step going non-finite while the
            # loss curve keeps reporting a stale number. Reject (not just clip) anything
            # past grad_skip_mult x grad_clip up front.
            grad_norm_ok = grad_norm_finite and grad_norm <= cfg["grad_clip"] * grad_skip_mult
            if grad_norm_ok:
                scaler.step(opt)
                grad_norm_sum += grad_norm.item()
                grad_norm_max = max(grad_norm_max, grad_norm.item())
            elif grad_norm_finite:
                n_skipped_large += 1
                print(f"  [skip] grad norm {grad_norm.item():.3e} > {grad_skip_mult}x grad_clip"
                      f" at epoch {epoch+1} -- step skipped")
            else:
                n_skipped += 1
                print(f"  [skip] non-finite grad norm ({grad_norm.item()}) at epoch {epoch+1} -- step skipped")
            scaler.update()
            opt.zero_grad()

            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        n_finite = len(train_loader) - n_skipped - n_skipped_large
        grad_norm_mean = grad_norm_sum / n_finite if n_finite else float("nan")

        # Logged every epoch, not gated by log_every, so a post-hoc wandb look can
        # pinpoint exactly which epoch an instability starts at instead of only
        # seeing the aftermath at the next checkpoint.
        if wandb.run is not None:
            wandb.log({"grad_norm_mean": grad_norm_mean, "grad_norm_max": grad_norm_max,
                       "n_skipped": n_skipped, "n_skipped_large": n_skipped_large}, step=epoch + 1)

        if (epoch + 1) % cfg['log_every'] == 0:
            log_dict = {"train_loss": avg_loss, "learning_rate": sched.get_last_lr()[0]}
            msg = (f"Epoch {epoch+1:4d}/{cfg['epochs']}  |  loss={avg_loss:.5f}"
                   f"  grad_norm(mean/max)={grad_norm_mean:.3f}/{grad_norm_max:.3f}")
            if n_skipped or n_skipped_large:
                msg += f"  (skipped {n_skipped} non-finite, {n_skipped_large} outlier-magnitude steps)"

            if test_loader is not None:
                model.eval()
                test_epoch_loss = 0.0
                with torch.no_grad():
                    for x1, cond, phi, theta, geom_id in test_loader:
                        x1, cond   = x1.to(device), cond.to(device)
                        phi, theta = phi.to(device), theta.to(device)
                        geom_id    = geom_id.to(device)

                        with torch.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                            loss = flow.training_loss(model, x1, cond, phi, theta, geom_id)

                        test_epoch_loss += loss.item()

                avg_test_loss = test_epoch_loss / len(test_loader)
                test_loss_history.append((epoch + 1, avg_test_loss))
                log_dict["test_loss"] = avg_test_loss
                msg += f"  test_loss={avg_test_loss:.5f}"

            msg += f"  lr={sched.get_last_lr()[0]:.2e}"
            print(msg)
            if wandb.run is not None:
                wandb.log(log_dict, step=epoch + 1)

        if (epoch + 1) % cfg["save_every"] == 0:
            # avg_loss can still read finite even after an Adam moment has gone
            # NaN/Inf (the poisoned moment silently corrupts every future update
            # while per-batch loss.item() stays a normal float) -- check the actual
            # parameters too, not just the loss, before writing a checkpoint.
            finite_weights = all(torch.isfinite(p).all() for p in model.parameters())
            if not math.isfinite(avg_loss) or not finite_weights:
                print(f"  -> epoch {epoch+1}: non-finite loss or weights, skipping checkpoint save")
            else:
                ckpt_path = ckpt_dir / f"ckpt{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}_epoch{epoch+1:04d}.pt"
                torch.save({
                    "epoch":  epoch,
                    "model":  model.state_dict(),
                    "opt":    opt.state_dict(),
                    "sched":  sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "cfg":    cfg,
                }, ckpt_path)
                print(f"  -> saved {ckpt_path}")

    return model, flow, loss_history, test_loss_history


# ── Local checkpoint / generation helpers ──────────────────────────────────────
# Not added to ../utilities.py: those DDIM equivalents (load_model_from_checkpoint,
# generate) are hard-wired to UNet3D/DDIMScheduler kwargs and would need a parallel
# code path anyway, so this model gets its own small local versions instead.

def load_fm_checkpoint(cfg: dict, model_cls, flow_cls):
    device   = cfg["device"]
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpts    = sorted(ckpt_dir.glob("ckpt*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    state      = torch.load(ckpts[-1], map_location=device)
    loaded_cfg = state["cfg"]
    print(f"Loading {ckpts[-1]}  (epoch {state['epoch'] + 1})")
    print(loaded_cfg)

    model = model_cls(
        voxel_shape  = loaded_cfg["voxel_shape"],
        patch_size   = loaded_cfg["patch_size"],
        embed_dim    = loaded_cfg["embed_dim"],
        depth        = loaded_cfg["depth"],
        num_heads    = loaded_cfg["num_heads"],
        mlp_ratio    = loaded_cfg["mlp_ratio"],
        cond_dim     = loaded_cfg["cond_dim"],
        n_geometries = loaded_cfg["n_geometries"],
        dropout      = loaded_cfg["dropout"],
    ).to(device)

    # Checkpoints saved from a torch.compile-wrapped model (cfg["compile"]=True)
    # have every key prefixed "_orig_mod." -- strip it so this loads regardless of
    # whether the checkpoint being loaded was saved compiled or not.
    state_dict = state["model"]
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    flow = flow_cls(device=device)

    n_params     = sum(p.numel() for p in model.parameters())
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"  Parameters : {n_params / 1e6:.2f} M  |  VRAM : {(param_bytes + buffer_bytes) / 1024**3:.3f} GiB")
    print(f"Model loaded from epoch {state['epoch'] + 1}.")
    return model, flow


def generate_fm(model, flow, cfg: dict, inverse, n_samples: int = 4,
                 e_inc_gev: float = 10.0, phi: float = 0.0, theta: float = math.pi / 2,
                 geom_id: int = 0, steps: int = None, method: str = None):
    """Generate shower samples conditioned on (energy, phi, theta, geometry).

    Parameters mirror ../utilities.py's generate(), but integrate the flow-matching
    ODE (steps/method) instead of DDIM's discrete reverse chain (ddim_steps/eta), and
    add angle/geometry conditioning. Defaults (phi=0, theta=pi/2, geom_id=0) match
    CaloChallenge Dataset 2's fixed incidence; pass geom_id=1 + real phi/theta for
    Par04SiW, with ``inverse=inverses[1]``. geom_id=2 + real phi/theta + inverse=
    inverses[2] generates against the held-out FCCeeALLEGRO geometry -- never
    trained on, so this is a zero-shot generalisation check, not a fit.
    """
    model.eval()
    device  = cfg["device"]
    D, H, W = cfg["voxel_shape"]
    cond     = torch.full((n_samples,), math.log10(e_inc_gev * 1e3), device=device)
    phi_t    = torch.full((n_samples,), phi, device=device)
    theta_t  = torch.full((n_samples,), theta, device=device)
    geom_t   = torch.full((n_samples,), geom_id, dtype=torch.long, device=device)
    shape    = (n_samples, 1, D, H, W)
    steps    = cfg["sampling_steps"]  if steps  is None else steps
    method   = cfg["sampling_method"] if method is None else method
    with torch.no_grad():
        samples = flow.sample(model, shape, cond, phi_t, theta_t, geom_t, steps=steps, method=method)
    return inverse(samples.clamp(-1, 1).cpu().numpy())
