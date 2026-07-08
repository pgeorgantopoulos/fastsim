import math
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ─── I/O ──────────────────────────────────────────────────────────────────────

class LEMURS(Dataset):
    """Raw calorimeter-shower dataset — no preprocessing.

    Loads and reshapes voxels; optionally applies a model-specific transform.
    Optionally also loads per-sample incident angle (``phi_key``/``theta_key``) —
    used for LEMURS files (e.g. FCCeeALLEGRO) that vary incidence angle, unlike
    CaloChallenge Dataset 2's fixed-angle showers.

    Parameters
    ----------
    transform : callable, optional
        Applied in __getitem__. Without angle keys:
        ``(shower_np, energy_mev) -> (x_tensor, cond_tensor)``.
        With angle keys: ``(shower_np, energy_mev, phi, theta) -> (...)``.
        When None, returns the corresponding raw tensor tuple.

    Attributes
    ----------
    showers  : ndarray (N, D, H, W) float32  — raw MeV deposits
    energies : ndarray (N,) float32          — incident energy in MeV
    phi, theta : ndarray (N,) float32 or None — incident angle in radians, if loaded
    """

    def __init__(self, path: str, showers_key: str, energies_key: str,
                 voxel_shape: tuple, data_slice=None, transform=None,
                 phi_key: str = None, theta_key: str = None):
        super().__init__()
        self.voxel_shape = voxel_shape
        self.transform   = transform

        sl = data_slice if data_slice is not None else ()
        with h5py.File(path, "r") as f:
            raw_showers  = f[showers_key][sl]
            raw_energies = f[energies_key][sl]
            raw_phi      = f[phi_key][sl]   if phi_key   is not None else None
            raw_theta    = f[theta_key][sl] if theta_key is not None else None

        self.showers  = raw_showers.astype(np.float32).reshape(-1, *voxel_shape)
        self.energies = raw_energies.astype(np.float32).flatten()
        self.phi      = raw_phi.astype(np.float32).flatten()   if raw_phi   is not None else None
        self.theta    = raw_theta.astype(np.float32).flatten() if raw_theta is not None else None

    @classmethod
    def from_config(cls, cfg: dict, train_split: float = 0.8, transform=None,
                     phi_key: str = None, theta_key: str = None):
        """Return ``(train_ds, test_ds)`` built from a config dict.

        Parameters
        ----------
        cfg : dict
            Must contain: ``data_path``, ``showers_key``, ``energies_key``,
            ``voxel_shape``.  Optional: ``n_samples`` (int, caps the dataset).
        train_split : float
            Fraction of samples used for training.
        transform : callable or None
            Passed to both datasets; can be replaced on each independently
            after construction (``ds.transform = ...``).
        phi_key, theta_key : str or None
            HDF5 keys for per-sample incident angle; omit for fixed-angle datasets.
        """
        path = cfg["train_data_path"] if "train_data_path" in cfg else cfg["data_path"]
        with h5py.File(path, "r") as f:
            total = len(f[cfg["showers_key"]]) if cfg.get("n_samples") is None else cfg["n_samples"]

        train_size = int(train_split * total)
        train_ds = cls(path, cfg["showers_key"], cfg["energies_key"],
                       cfg["voxel_shape"], data_slice=slice(0, train_size),
                       transform=transform, phi_key=phi_key, theta_key=theta_key)
        test_ds  = cls(path, cfg["showers_key"], cfg["energies_key"],
                       cfg["voxel_shape"], data_slice=slice(train_size, total),
                       transform=transform, phi_key=phi_key, theta_key=theta_key)

        print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Voxel shape: {cfg['voxel_shape']}")
        return train_ds, test_ds

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, idx):
        shower = self.showers[idx]       # (D, H, W)
        energy = self.energies[idx]      # scalar MeV

        if self.phi is not None:
            phi, theta = self.phi[idx], self.theta[idx]
            if self.transform is not None:
                return self.transform(shower, energy, phi, theta)
            return (torch.from_numpy(shower), torch.tensor(energy, dtype=torch.float32),
                    torch.tensor(phi, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32))

        if self.transform is not None:
            return self.transform(shower, energy)
        return torch.from_numpy(shower), torch.tensor(energy, dtype=torch.float32)


class CaloChallenge(Dataset):
    """CaloChallenge shower dataset reshaped to (R, PHI, Z) per sample.

    Parameters
    ----------
    filename : str
        Path to the HDF5 file.
    hlf : HighLevelFeatures
        Initialised HLF object matching the dataset's binning XML.
    i_idcs : array-like or None
        Row indices into the HDF5 file; None loads all rows.
    transform : callable(shower_np, energy_mev) -> (x_tensor, cond_tensor), optional
        Applied in __getitem__.  When None, returns raw
        ``(Tensor(R,PHI,Z) float32, Tensor() float32 in MeV)``.

    Attributes
    ----------
    showers  : ndarray (N, R, PHI, Z) float32
    energies : ndarray (N,) float32
    """

    def __init__(self, filename: str, hlf, i_idcs=None, transform=None):
        super().__init__()
        self.transform = transform

        with h5py.File(filename, "r") as f:
            print("Features in " + os.path.basename(filename) + ":", list(f.keys()))
            showers           = np.array(f["showers"][i_idcs, :] if i_idcs is not None else f["showers"][:])
            incident_energies = np.array(f["incident_energies"][i_idcs] if i_idcs is not None else f["incident_energies"][:])

        print("Showers shape:", showers.shape)
        print("Incident energies shape:", incident_energies.shape)

        layer_bounds = np.unique(hlf.bin_edges)
        num_layers   = len(hlf.relevantLayers)
        max_num_rad  = max(len(r) - 1 for r in hlf.r_edges)
        max_num_ang  = max(hlf.num_alpha)
        K            = showers.shape[0]

        showers_4d = np.zeros((K, max_num_rad, max_num_ang, num_layers), dtype=showers.dtype)
        for idx, _ in enumerate(hlf.relevantLayers):
            n_ang      = hlf.num_alpha[idx]
            n_rad      = len(hlf.r_edges[idx]) - 1
            layer_data = showers[:, layer_bounds[idx]:layer_bounds[idx + 1]]    # (K, n_ang*n_rad)
            layer_data = layer_data.reshape(K, n_ang, n_rad)                    # (K, n_ang, n_rad)
            showers_4d[:, :n_rad, :n_ang, idx] = layer_data.transpose(0, 2, 1) # (K, n_rad, n_ang)

        print("showers_4d shape:", showers_4d.shape)
        self.showers  = showers_4d.astype(np.float32)
        self.energies = incident_energies.astype(np.float32).flatten()

    @classmethod
    def from_config(cls, cfg: dict, hlf, train_split: float = 0.8, transform=None):
        """Return ``(train_ds, test_ds)`` built from a config dict.

        Parameters
        ----------
        cfg : dict
            Must contain: ``data_path``.  Optional: ``n_samples`` (int, caps the dataset).
        hlf : HighLevelFeatures
            Passed through to both dataset instances.
        train_split : float
            Fraction of samples used for training.
        transform : callable or None
            Passed to both datasets; can be replaced on each independently
            after construction (``ds.transform = ...``).
        """
        path = cfg["train_data_path"] if "train_data_path" in cfg else cfg["data_path"]
        with h5py.File(path, "r") as f:
            total = len(f["showers"]) if cfg.get("n_samples") is None else cfg["n_samples"]

        idcs       = np.arange(total)
        train_size = int(train_split * total)
        train_ds   = cls(path, hlf, i_idcs=idcs[:train_size], transform=transform)
        test_ds    = cls(path, hlf, i_idcs=idcs[train_size:], transform=transform)

        print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
        return train_ds, test_ds

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, idx):
        shower = self.showers[idx]   # (R, PHI, Z)
        energy = self.energies[idx]  # scalar MeV
        if self.transform is not None:
            return self.transform(shower, energy)
        return torch.from_numpy(shower), torch.tensor(energy, dtype=torch.float32)


# ─── Data loading ─────────────────────────────────────────────────────────────

def ddim_calochallenge_dataloaders(cfg: dict, hlf, log_eps: float = 1e-6,
                                   train_split: float = 0.8):
    """Build CaloChallenge dataloaders with log-normalisation fitted on training data.

    Parameters
    ----------
    cfg : dict
        Must contain ``data_path`` and ``batch_size``; optional ``n_samples``.
    hlf : HighLevelFeatures
        Initialised HLF object matching the dataset's binning XML.

    Returns
    -------
    train_loader, test_loader, inverse
        ``inverse(x_norm)`` maps a [-1, 1] array/tensor back to raw MeV deposits.
    """
    train_ds, test_ds = CaloChallenge.from_config(cfg, hlf, train_split=train_split)

    logged = np.log(train_ds.showers + log_eps)
    vmin   = float(logged.min())
    vmax   = float(logged.max())

    def forward(shower: np.ndarray, energy_mev: float):
        shower = shower.transpose(2, 1, 0)   # (R, PHI, Z) → (Z, PHI, R) = voxel_shape order
        x    = (np.log(shower + log_eps) - vmin) / (vmax - vmin) * 2 - 1
        cond = math.log10(float(energy_mev) + 1e-9)
        return torch.from_numpy(x).unsqueeze(0).float(), torch.tensor(cond, dtype=torch.float32)

    def inverse(x_norm: np.ndarray) -> np.ndarray:
        return np.exp((x_norm + 1) / 2 * (vmax - vmin) + vmin) - log_eps

    train_ds.transform = test_ds.transform = forward

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)
    return train_loader, test_loader, inverse


def multi_geometry_dataloaders(cfg: dict, hlf, log_eps: float = 1e-6, train_split: float = 0.8):
    """Build a combined dataloader across two real geometries:
    CaloChallenge Dataset 2 (SiW, ``geom_id=0``, fixed incidence) and LEMURS
    FCCeeALLEGRO (``geom_id=1``, variable incidence angle).

    Both are reshaped to the same ``(Z, PHI, R) = voxel_shape`` axis order and
    log-normalised to ``[-1, 1]`` with their *own* fitted min/max — different
    detectors/absorbers have different deposit-energy scales, so a shared
    normalisation would over/under-saturate one of them.

    Parameters
    ----------
    cfg : dict
        Must contain: ``calo_train_data_path`` (internally 80/20-split into
        CaloChallenge train/held-out slices by ``CaloChallenge.from_config`` —
        ``calo_test_data_path`` is *not* read by this function, only by eval code
        that pulls ground-truth showers from that separate file directly),
        ``lemurs_data_path``, ``showers_key`` (shared — both files use ``"showers"``),
        ``lemurs_energies_key`` (LEMURS uses ``"incident_energy"``, singular — distinct
        from CaloChallenge's hardcoded ``"incident_energies"``), ``lemurs_phi_key``,
        ``lemurs_theta_key``, ``voxel_shape`` (CaloChallenge/model convention, e.g.
        ``(45, 16, 9)``), ``batch_size``. Optional: ``n_samples``.
    hlf : HighLevelFeatures
        Initialised HLF object matching the CaloChallenge binning.

    Returns
    -------
    train_loader, test_loader, inverses
        Each batch is ``(x, cond, phi, theta, geom_id)``. ``inverses`` is a dict
        ``{geom_id: inverse_fn}``, each mapping a ``[-1, 1]`` array back to raw
        MeV deposits for that geometry.
    """
    THETA_FIXED = math.pi / 2   # CaloChallenge Dataset 2: perpendicular incidence
    PHI_FIXED   = 0.0           # azimuthally symmetric — arbitrary but fixed

    # --- CaloChallenge Dataset 2 (SiW) — geom_id = 0 ---------------------------
    calo_cfg = {**cfg, "train_data_path": cfg["calo_train_data_path"],
                "test_data_path": cfg["calo_test_data_path"]}
    calo_train, calo_test = CaloChallenge.from_config(calo_cfg, hlf, train_split=train_split)

    calo_logged = np.log(calo_train.showers + log_eps)
    calo_vmin, calo_vmax = float(calo_logged.min()), float(calo_logged.max())

    def calo_forward(shower: np.ndarray, energy_mev: float):
        shower = shower.transpose(2, 1, 0)   # (R, PHI, Z) → (Z, PHI, R)
        x    = (np.log(shower + log_eps) - calo_vmin) / (calo_vmax - calo_vmin) * 2 - 1
        cond = math.log10(float(energy_mev) + 1e-9)
        return (torch.from_numpy(x).unsqueeze(0).float(), torch.tensor(cond, dtype=torch.float32),
                torch.tensor(PHI_FIXED, dtype=torch.float32), torch.tensor(THETA_FIXED, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long))

    def calo_inverse(x_norm: np.ndarray) -> np.ndarray:
        return np.exp((x_norm + 1) / 2 * (calo_vmax - calo_vmin) + calo_vmin) - log_eps

    calo_train.transform = calo_test.transform = calo_forward

    # --- LEMURS FCCeeALLEGRO — geom_id = 1 -------------------------------------
    lemurs_raw_shape = tuple(reversed(cfg["voxel_shape"]))   # (45,16,9) -> (9,16,45) = (R,PHI,Z)
    lemurs_cfg = {**cfg, "train_data_path": cfg["lemurs_data_path"], "voxel_shape": lemurs_raw_shape,
                  "energies_key": cfg["lemurs_energies_key"]}
    lemurs_train, lemurs_test = LEMURS.from_config(
        lemurs_cfg, train_split=train_split,
        phi_key=cfg["lemurs_phi_key"], theta_key=cfg["lemurs_theta_key"],
    )

    lemurs_logged = np.log(lemurs_train.showers + log_eps)
    lem_vmin, lem_vmax = float(lemurs_logged.min()), float(lemurs_logged.max())

    def lemurs_forward(shower: np.ndarray, energy_mev: float, phi: float, theta: float):
        shower = shower.transpose(2, 1, 0)   # (R, PHI, Z) → (Z, PHI, R)
        x    = (np.log(shower + log_eps) - lem_vmin) / (lem_vmax - lem_vmin) * 2 - 1
        cond = math.log10(float(energy_mev) + 1e-9)
        return (torch.from_numpy(x).unsqueeze(0).float(), torch.tensor(cond, dtype=torch.float32),
                torch.tensor(float(phi), dtype=torch.float32), torch.tensor(float(theta), dtype=torch.float32),
                torch.tensor(1, dtype=torch.long))

    def lemurs_inverse(x_norm: np.ndarray) -> np.ndarray:
        return np.exp((x_norm + 1) / 2 * (lem_vmax - lem_vmin) + lem_vmin) - log_eps

    lemurs_train.transform = lemurs_test.transform = lemurs_forward

    train_ds = torch.utils.data.ConcatDataset([calo_train, lemurs_train])
    test_ds  = torch.utils.data.ConcatDataset([calo_test,  lemurs_test])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)

    inverses = {0: calo_inverse, 1: lemurs_inverse}
    return train_loader, test_loader, inverses


# ─── Checkpointing ────────────────────────────────────────────────────────────

def load_model_from_checkpoint(cfg: dict, model_cls, scheduler_cls):
    """Load the latest checkpoint; return a ready-to-eval (model, scheduler) pair.

    Parameters
    ----------
    cfg : dict
        Must contain ``ckpt_dir`` and ``device``.
    model_cls : type
        Constructor called with kwargs from the saved cfg:
        ``voxel_shape``, ``base_ch``, ``ch_mults``, ``attn_res``, ``cond_dim``.
    scheduler_cls : type
        Constructor called with kwargs ``T``, ``beta_min``, ``beta_max``,
        ``schedule``, ``device``.
    """
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
        voxel_shape = loaded_cfg["voxel_shape"],
        base_ch     = loaded_cfg["base_ch"],
        ch_mults    = loaded_cfg["ch_mults"],
        attn_res    = loaded_cfg["attn_res"],
        cond_dim    = loaded_cfg["cond_dim"],
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    scheduler = scheduler_cls(
        T        = loaded_cfg["T"],
        beta_min = loaded_cfg["beta_min"],
        beta_max = loaded_cfg["beta_max"],
        schedule = loaded_cfg["schedule"],
        device   = device,
    )

    n_params     = sum(p.numel() for p in model.parameters())
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    print(f"  Parameters : {n_params / 1e6:.2f} M  |  VRAM : {(param_bytes + buffer_bytes) / 1024**3:.3f} GiB")
    print(f"Model loaded from epoch {state['epoch'] + 1}.")
    return model, scheduler


# ─── Inference ────────────────────────────────────────────────────────────────

def generate(model, scheduler, cfg: dict, inverse, n_samples: int = 4,
             e_inc_gev: float = 10.0, ddim_steps: int = 50, eta: float = 0.0):
    """Generate shower samples conditioned on a fixed incident energy.

    Parameters
    ----------
    model : nn.Module
    scheduler : DDIMScheduler-like — must implement ``ddim_sample``.
    cfg : dict — must contain ``device`` and ``voxel_shape``.
    inverse : callable — maps [-1, 1] array back to MeV deposits.
    n_samples : int
    e_inc_gev : float — incident energy in GeV.
    ddim_steps : int
    eta : float — 0 = deterministic DDIM, 1 = DDPM-equivalent stochasticity.

    Returns
    -------
    ndarray (N, 1, D, H, W) in MeV.
    """
    model.eval()
    device  = cfg["device"]
    D, H, W = cfg["voxel_shape"]
    cond    = torch.full((n_samples,), math.log10(e_inc_gev * 1e3), device=device)
    shape   = (n_samples, 1, D, H, W)
    with torch.no_grad():
        samples = scheduler.ddim_sample(model, shape, cond, steps=ddim_steps, eta=eta)
    return inverse(samples.clamp(-1, 1).cpu().numpy())


# ─── Shared visualisation ──────────────────────────────────────────────────────

def plot_shower_3d(shower3d, i_idx=0, incident_energy=None, threshold=1):
    """Plotly 3D scatter of a single shower in cylindrical (R, PHI, Z) coordinates.

    Parameters
    ----------
    shower3d : ndarray (R, PHI, Z)
    i_idx : int
        Event index used only for the plot title.
    incident_energy : float or None
        Incident energy in MeV; shown in title when provided.
    threshold : float
        Voxels below this value (MeV) are hidden.
    """
    num_rad_splits, num_ang_splits, num_layers = shower3d.shape

    R   = np.arange(num_rad_splits)
    PHI = np.linspace(0, 2 * np.pi, num_ang_splits, endpoint=False)
    Z   = np.arange(num_layers)

    r, phi, z = np.meshgrid(R, PHI, Z, indexing='ij')
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    mask = shower3d.flatten() >= threshold
    vals = shower3d.flatten()[mask]

    title_text = f"Shower #{i_idx}"
    if incident_energy is not None:
        title_text += f" — E<sub>inc</sub> = {float(np.squeeze(incident_energy)) / 1e3:.2f} GeV"
    title_text += (f"  | max(E) = {shower3d.max():.2f} MeV"
                   f" | threshold = {threshold / 1e3} GeV")

    fig = go.Figure(data=[
        go.Scatter3d(
            x=x.flatten()[mask],
            y=y.flatten()[mask],
            z=z.flatten()[mask],
            mode='markers',
            marker=dict(
                size=5,
                opacity=1,
                colorscale='greens',
                color=np.log(vals + 1),
                cmin=0,
                cmax=np.log(shower3d.max() + 1),
                symbol='square',
                showscale=True,
                colorbar=dict(title="log(MeV+1)", thickness=8, len=0.35, x=1.0),
            ),
        ),
        go.Scatter3d(
            x=[x.flatten().min(), x.flatten().max()],
            y=[0, 0], z=[0, 0],
            mode='lines', line=dict(color='red', width=4), name='X-axis',
        ),
        go.Scatter3d(
            x=[0, 0],
            y=[y.flatten().min(), y.flatten().max()], z=[0, 0],
            mode='lines', line=dict(color='blue', width=4), name='Y-axis',
        ),
        go.Scatter3d(
            x=[0, 0], y=[0, 0],
            z=[z.flatten().min(), z.flatten().max()],
            mode='lines', line=dict(color='green', width=4), name='Z-axis',
        ),
    ])

    fig.update_layout(
        title=dict(text=title_text, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="rgba(0,0,0,0)",
                       showbackground=True, gridcolor="lightgray"),
            yaxis=dict(title="Y", backgroundcolor="rgba(0,0,0,0)",
                       showbackground=True, gridcolor="lightgray"),
            zaxis=dict(title="Z", backgroundcolor="rgba(0,0,0,0)",
                       showbackground=True, gridcolor="lightgray"),
            camera=dict(
                up=dict(x=0, y=0.1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.5, y=1.5, z=-1.8),
            ),
        ),
        height=1000,
        width=1500,
    )

    return fig


def plot_z_profiles(showers, gen_showers=None, filename='', i_idcs=None,
                    transverse_axes=(1, 2)):
    """Plot depth profiles (summed over transverse axes) for one or two shower batches.

    Parameters
    ----------
    showers : ndarray (K, ...)       — reference batch.
    gen_showers : ndarray or None    — generated batch; when given, draws a comparison
                                       instead of the single-batch lin/log view.
    filename : str                   — used in figure title.
    i_idcs : int or None             — max showers to overlay (single-batch mode only).
    transverse_axes : tuple          — axes to sum over for the depth profile.
        ``(1, 2)`` for CaloChallenge layout (K, R, PHI, Z);
        ``(2, 3)`` for LEMURS / DDIM layout (N, D, H, W).
    """
    z_profiles = showers.sum(axis=transverse_axes)

    if gen_showers is not None:
        gen_profiles = gen_showers.sum(axis=transverse_axes)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(z_profiles.T, lw=0.1, color='k', linestyle='--', label='Reference')
        ax.plot(gen_profiles.T, lw=0.1, color='r', label='Generated')
        title = 'Depth Profiles: Reference vs Generated'
        if filename:
            title += ' | ' + filename
        ax.set_title(title)
        ax.set_xlabel('Depth Layer')
        ax.set_ylabel('Total Energy (MeV)')
        ax.grid(True, linestyle='--', alpha=0.7)
        # ax.legend()
        return fig, ax

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle('Z-profiles: ' + filename + ' | slice: ' + str(i_idcs))
    ax[0].plot(z_profiles[:i_idcs].T, 'k-', lw=0.1)
    ax[1].semilogy(z_profiles[:i_idcs].T, 'k-', lw=0.1)
    for a in ax:
        a.grid('auto')
    ax[0].set_title('MeV')
    ax[1].set_title('log(MeV)')
    return fig, ax


# ─── LEMURS-specific visualisation ────────────────────────────────────────────

def plot_layer_mesh3d(shower3d, layers=None, n_cols=5):
    """Plotly 3D Mesh subplot grid — one panel per calorimeter layer.

    Parameters
    ----------
    shower3d : ndarray (R, PHI, Z)
        Single shower.
    layers : list[int] or None
        Layer indices to display; defaults to all layers.
    n_cols : int
        Number of columns in the subplot grid.
    """
    num_rad_splits, num_ang_splits, num_layers = shower3d.shape

    R   = np.arange(num_rad_splits)
    PHI = np.linspace(0, 2 * np.pi, num_ang_splits, endpoint=False)
    r, phi = np.meshgrid(R, PHI, indexing='ij')
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    if layers is None:
        layers = list(range(num_layers))

    n_rows = (len(layers) + n_cols - 1) // n_cols
    specs  = [[{'type': 'scene'} for _ in range(n_cols)] for _ in range(n_rows)]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=[f"Layer {i}" for i in layers],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    for idx, layer_idx in enumerate(layers):
        row      = idx // n_cols + 1
        col      = idx % n_cols + 1
        shower2d = shower3d[:, :, layer_idx]

        fig.add_trace(
            go.Mesh3d(
                x=x.flatten(),
                y=y.flatten(),
                z=shower2d.flatten(),
                intensity=shower2d.flatten(),
                opacity=0.7,
                showscale=(idx == len(layers) - 1),
                colorbar=dict(
                    title="Energy (MeV)",
                    orientation="h",
                    x=0.5, y=-0.2,
                    xanchor="center",
                    yanchor="top",
                ),
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=300 * n_rows,
        width=300 * n_cols,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    for idx in range(1, len(layers) + 1):
        fig.update_layout({
            f'scene{idx if idx > 1 else ""}': dict(
                xaxis=dict(title='', showticklabels=False),
                yaxis=dict(title='', showticklabels=False),
                zaxis=dict(title='', showticklabels=False),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.8),
            )
        })

    return fig


# ─── CaloChallenge-specific visualisation ─────────────────────────────────────

def plot_single_shower_layers(showers, hlf, i_idx=0):
    """Draw per-layer profiles of a single flat CaloChallenge shower.

    Parameters
    ----------
    showers : ndarray (K, V)
        Flat shower array.
    hlf : HighLevelFeatures
    i_idx : int
        Event index.

    Returns (fig_per_layer, fig_full).
    """
    shower    = showers[i_idx, :]
    hlf.DrawSingleShower(shower)

    num_layers = len(hlf.relevantLayers)
    lr_bound   = np.unique(hlf.bin_edges)

    fig1, axes = plt.subplots(1, num_layers, figsize=(20, 4))
    for lr_idx in range(num_layers):
        axes[lr_idx].plot(showers[i_idx, lr_bound[lr_idx]:lr_bound[lr_idx + 1]])

    fig2, ax2 = plt.subplots(figsize=(20, 4))
    ax2.plot(shower)
    for lr_idx in range(num_layers):
        ax2.axvline(x=lr_bound[lr_idx], color='red', linestyle='--')
    ax2.set_xlim(0, np.array(shower.shape).prod())

    return fig1, fig2


def plot_average_shower(showers, incident_energies, hlf, nrg_lvl=None):
    """Draw average and variance shower profiles for a given energy level.

    Parameters
    ----------
    showers : ndarray (K, V)
        Flat shower array.
    incident_energies : ndarray (K,) or (K, 1)
    hlf : HighLevelFeatures
    nrg_lvl : float or None
        Energy level in MeV; defaults to the lowest unique level.

    Returns (fig_per_layer, fig_mean, fig_var).
    """
    incident_energies = incident_energies.flatten()
    nrg_lvls = np.unique(incident_energies)
    if nrg_lvl is None:
        nrg_lvl = nrg_lvls[0]

    i_idx = np.where(incident_energies == nrg_lvl)[0]
    hlf.DrawAverageShower(showers[i_idx])

    num_layers = len(hlf.relevantLayers)
    lr_bound   = np.unique(hlf.bin_edges)

    fig1, axes = plt.subplots(1, num_layers, figsize=(20, 4))
    for lr_idx in range(num_layers):
        layer_mean = np.mean(showers[i_idx, lr_bound[lr_idx]:lr_bound[lr_idx + 1]], axis=0)
        axes[lr_idx].plot(layer_mean)

    vox   = showers[i_idx, :]
    n_vox = showers.shape[-1]

    fig2, ax2 = plt.subplots(figsize=(20, 4))
    ax2.semilogy(np.mean(vox, axis=0), label="mean")
    for lr_idx in range(num_layers):
        ax2.axvline(x=lr_bound[lr_idx], color='red', linestyle='--')
    ax2.set_xlim(0, n_vox)
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(20, 4))
    ax3.semilogy(np.var(vox, axis=0), 'r', alpha=0.8, label='var')
    for lr_idx in range(num_layers):
        ax3.axvline(x=lr_bound[lr_idx], color='red', linestyle='--')
    ax3.set_xlim(0, n_vox)
    ax3.legend()

    return fig1, fig2, fig3


def plot_comparison(ref_showers, ref_energies, gen_showers, gen_energies,
                    gen_label='Model', n_cols=3):
    """Energy-distribution and radial-profile comparison with ratio panels and EMD scores.

    Parameters
    ----------
    ref_showers : ndarray (K, R, PHI, Z)   Geant4 reference.
    ref_energies : ndarray (K,) or (K, 1)
    gen_showers : ndarray (K, R, PHI, Z)   Model output.
    gen_energies : ndarray (K,) or (K, 1)
    gen_label : str
    n_cols : int
        Number of energy levels shown side by side.

    Returns fig.
    """
    ref_energies = ref_energies.flatten()
    gen_energies = gen_energies.flatten()

    unique_e = np.unique(ref_energies)
    e_levels = unique_e[np.linspace(0, len(unique_e) - 1, n_cols, dtype=int)]

    def _histo_ratio(ax_top, ax_bot, ref_vals, gen_vals, bins, xlabel, ylabel, lbl):
        eps = 1e-8
        c_ref, edges, _ = ax_top.hist(ref_vals, bins=bins, label='Geant4',
                                       color='silver', histtype='stepfilled')
        emd = wasserstein_distance(ref_vals, gen_vals) if len(gen_vals) > 0 else np.nan
        c_gen, _, _     = ax_top.hist(gen_vals, bins=bins,
                                       label=f'{lbl} (EMD: {emd:.4f})',
                                       color='steelblue', histtype='step', linewidth=1.5)
        ax_top.set_yscale('log')
        ax_top.set_ylabel(ylabel)
        ax_top.legend(fontsize=7)
        ax_top.tick_params(labelbottom=False)

        xc    = 0.5 * (edges[:-1] + edges[1:])
        ratio = c_gen / (c_ref + eps)
        ax_bot.axhline(1.0, color='k', linestyle='--', linewidth=0.8)
        ax_bot.plot(xc, ratio, 'o', color='steelblue', markersize=2)
        ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_ylabel('Ratio')
        ax_bot.set_xlabel(xlabel)

    fig   = plt.figure(figsize=(6 * n_cols, 15))
    outer = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.5, wspace=0.35)

    for col, E in enumerate(e_levels):
        ref_mask = np.isclose(ref_energies, E, rtol=1e-3)
        gen_mask = np.isclose(gen_energies, E, rtol=1e-3)
        ref_sel  = ref_showers[ref_mask]
        gen_sel  = gen_showers[gen_mask]

        # Energy distribution
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, col], height_ratios=[3, 1], hspace=0.05)
        ax0 = fig.add_subplot(inner[0])
        ax1 = fig.add_subplot(inner[1], sharex=ax0)

        ref_vox = ref_sel.flatten()
        gen_vox = gen_sel.flatten() if len(gen_sel) > 0 else np.zeros(1)
        ref_nz  = ref_vox[ref_vox > 0]
        gen_nz  = gen_vox[gen_vox > 0] if gen_vox.any() else np.zeros(1)

        vmax   = max(ref_nz.max(), gen_nz.max()) if len(ref_nz) > 0 else 1.
        bins_e = np.linspace(0, vmax, 51)

        ax0.set_title(f'Energy Distribution\n$E_{{inc}}$ = {E:.0f} MeV')
        _histo_ratio(ax0, ax1, ref_nz, gen_nz, bins_e,
                     'Deposited Energy [MeV]', 'Number of Hits', gen_label)

        # Radial profile
        inner2 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1, col], height_ratios=[3, 1], hspace=0.05)
        ax2 = fig.add_subplot(inner2[0])
        ax3 = fig.add_subplot(inner2[1], sharex=ax2)

        ref_rprof = ref_sel.sum(axis=(2, 3)).mean(axis=0)
        gen_rprof = (gen_sel.sum(axis=(2, 3)).mean(axis=0)
                     if len(gen_sel) > 0 else np.zeros_like(ref_rprof))

        n_r     = len(ref_rprof)
        r_edges = np.arange(n_r + 1) / n_r
        r_cent  = 0.5 * (r_edges[:-1] + r_edges[1:])

        ref_w = ref_rprof / (ref_rprof.sum() + 1e-16)
        gen_w = gen_rprof / (gen_rprof.sum() + 1e-16)
        emd_r = wasserstein_distance(r_cent, r_cent, ref_w, gen_w)

        ax2.fill_between(r_edges, np.append(ref_rprof, ref_rprof[-1]),
                         step='post', color='silver', alpha=1., label='Geant4')
        ax2.step(r_edges, np.append(ref_rprof, ref_rprof[-1]),
                 where='post', color='gray', linewidth=0.5)
        ax2.step(r_edges, np.append(gen_rprof, gen_rprof[-1]),
                 where='post', color='steelblue', linewidth=1.5,
                 label=f'{gen_label} (EMD: {emd_r:.4f})')
        ax2.set_ylabel('Mean Deposited Energy [MeV]')
        ax2.legend(fontsize=7)
        ax2.tick_params(labelbottom=False)

        eps     = 1e-8
        ratio_r = gen_rprof / (ref_rprof + eps)
        ax3.axhline(1.0, color='k', linestyle='--', linewidth=0.8)
        ax3.plot(r_cent, ratio_r, 'o', color='steelblue', markersize=3)
        ax3.set_ylim(0.5, 1.5)
        ax3.set_ylabel('Ratio')
        ax3.set_xlabel('r-bin')

        # Longitudinal profile
        inner3 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[2, col], height_ratios=[3, 1], hspace=0.05)
        ax4 = fig.add_subplot(inner3[0])
        ax5 = fig.add_subplot(inner3[1], sharex=ax4)

        ref_lprof = ref_sel.sum(axis=(1, 2)).mean(axis=0)
        gen_lprof = (gen_sel.sum(axis=(1, 2)).mean(axis=0)
                     if len(gen_sel) > 0 else np.zeros_like(ref_lprof))

        n_z     = len(ref_lprof)
        z_edges = np.arange(n_z + 1)
        z_cent  = 0.5 * (z_edges[:-1] + z_edges[1:])

        ref_wz = ref_lprof / (ref_lprof.sum() + 1e-16)
        gen_wz = gen_lprof / (gen_lprof.sum() + 1e-16)
        emd_z  = wasserstein_distance(z_cent, z_cent, ref_wz, gen_wz)

        ax4.fill_between(z_edges, np.append(ref_lprof, ref_lprof[-1]),
                         step='post', color='silver', alpha=1., label='Geant4')
        ax4.step(z_edges, np.append(ref_lprof, ref_lprof[-1]),
                 where='post', color='gray', linewidth=0.5)
        ax4.step(z_edges, np.append(gen_lprof, gen_lprof[-1]),
                 where='post', color='steelblue', linewidth=1.5,
                 label=f'{gen_label} (EMD: {emd_z:.4f})')
        ax4.set_title('Longitudinal Profile', fontsize=8)
        ax4.set_ylabel('Mean Deposited Energy [MeV]')
        ax4.legend(fontsize=7)
        ax4.tick_params(labelbottom=False)

        ratio_z = gen_lprof / (ref_lprof + eps)
        ax5.axhline(1.0, color='k', linestyle='--', linewidth=0.8)
        ax5.plot(z_cent, ratio_z, 'o', color='steelblue', markersize=3)
        ax5.set_ylim(0.5, 1.5)
        ax5.set_ylabel('Ratio')
        ax5.set_xlabel('z-layer')

    plt.suptitle(f'Shower Comparison: Geant4 vs {gen_label}', fontsize=14, y=1.01)
    return fig


# ─── DDIM / LEMURS-specific visualisation ─────────────────────────────────────

def plot_layers(shower_data, cfg: dict, title: str = '', e_inc_gev=None, n_cols: int = 9):
    """Polar pcolormesh grid — one panel per calorimeter depth-layer.

    Parameters
    ----------
    shower_data : ndarray (D, H, W)  — single shower in (depth, phi, r) layout.
    cfg : dict                        — must contain ``voxel_shape``.
    title : str
    e_inc_gev : float or None        — appended to suptitle when given.
    n_cols : int                     — columns in the subplot grid.
    """
    D, H, W     = cfg["voxel_shape"]
    n_rows      = math.ceil(D / n_cols)
    theta_edges = np.linspace(0, 2 * np.pi, H + 1)
    rho_edges   = np.arange(W + 1)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                              figsize=(n_cols * 1.2, n_rows * 1.2),
                              subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    full_title = title + (f' (E_inc = {e_inc_gev:.0f} GeV)' if e_inc_gev is not None else '')
    fig.suptitle(full_title, fontsize=16)

    global_min, global_max = shower_data.min(), shower_data.max()
    mesh = None
    for i in range(D):
        ax   = axes[i]
        mesh = ax.pcolormesh(theta_edges, rho_edges, shower_data[i].T,
                             cmap='jet', vmin=global_min, vmax=global_max)
        ax.set_title(f'Layer {i}', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    for ax in axes[D:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    cbar = fig.colorbar(mesh, ax=axes[:D].tolist(), orientation='vertical',
                        shrink=0.75, pad=0.02)
    cbar.set_label('Energy (MeV)')
    return fig, axes


# ─── Validation Observables (CaloChallenge / CaloDiT-2 format) ────────────────
# Input convention: showers (K, R, PHI, Z)

def compute_long_total_energy(showers):
    """Mean energy deposited per Z-layer, averaged over events. Returns (Z,)."""
    return showers.sum(axis=(1, 2)).mean(axis=0)

def compute_long_total_hits(showers, threshold=0.0):
    """Mean hit count per Z-layer, averaged over events. Returns (Z,)."""
    return (showers > threshold).sum(axis=(1, 2)).mean(axis=0).astype(float)

def compute_long_first_moment(showers):
    """Energy-weighted mean Z index per event. Returns (K,)."""
    Z = showers.shape[3]
    z = np.arange(Z, dtype=np.float64)
    e_z = showers.sum(axis=(1, 2))
    return (e_z * z).sum(axis=1) / (e_z.sum(axis=1) + 1e-16)

def compute_long_second_moment(showers):
    """Energy-weighted second moment <z²> per event. Returns (K,)."""
    Z = showers.shape[3]
    z = np.arange(Z, dtype=np.float64)
    e_z = showers.sum(axis=(1, 2))
    return (e_z * z ** 2).sum(axis=1) / (e_z.sum(axis=1) + 1e-16)

def compute_long_event_energy(showers):
    """Energy per Z-layer per event. Returns (K, Z)."""
    return showers.sum(axis=(1, 2))

def compute_rad_total_energy(showers):
    """Mean energy deposited per R-bin, averaged over events. Returns (R,)."""
    return showers.sum(axis=(2, 3)).mean(axis=0)

def compute_rad_total_hits(showers, threshold=0.0):
    """Mean hit count per R-bin, averaged over events. Returns (R,)."""
    return (showers > threshold).sum(axis=(2, 3)).mean(axis=0).astype(float)

def compute_rad_first_moment(showers):
    """Energy-weighted mean R index per event. Returns (K,)."""
    R = showers.shape[1]
    r = np.arange(R, dtype=np.float64)
    e_r = showers.sum(axis=(2, 3))
    return (e_r * r).sum(axis=1) / (e_r.sum(axis=1) + 1e-16)

def compute_rad_second_moment(showers):
    """Energy-weighted second moment <r²> per event. Returns (K,)."""
    R = showers.shape[1]
    r = np.arange(R, dtype=np.float64)
    e_r = showers.sum(axis=(2, 3))
    return (e_r * r ** 2).sum(axis=1) / (e_r.sum(axis=1) + 1e-16)

def compute_rad_event_energy(showers):
    """Energy per R-bin per event. Returns (K, R)."""
    return showers.sum(axis=(2, 3))

def compute_azim_total_energy(showers):
    """Mean energy deposited per PHI-bin, averaged over events. Returns (PHI,)."""
    return showers.sum(axis=(1, 3)).mean(axis=0)

def compute_azim_total_hits(showers, threshold=0.0):
    """Mean hit count per PHI-bin, averaged over events. Returns (PHI,)."""
    return (showers > threshold).sum(axis=(1, 3)).mean(axis=0).astype(float)

def compute_azim_first_moment(showers):
    """Energy-weighted mean PHI index per event. Returns (K,)."""
    P = showers.shape[2]
    phi = np.arange(P, dtype=np.float64)
    e_phi = showers.sum(axis=(1, 3))
    return (e_phi * phi).sum(axis=1) / (e_phi.sum(axis=1) + 1e-16)

def compute_azim_second_moment(showers):
    """Energy-weighted second moment <φ²> per event. Returns (K,)."""
    P = showers.shape[2]
    phi = np.arange(P, dtype=np.float64)
    e_phi = showers.sum(axis=(1, 3))
    return (e_phi * phi ** 2).sum(axis=1) / (e_phi.sum(axis=1) + 1e-16)

def compute_azim_event_energy(showers):
    """Energy per PHI-bin per event. Returns (K, PHI)."""
    return showers.sum(axis=(1, 3))

def compute_total_event_energy(showers):
    """Total energy deposited per event. Returns (K,)."""
    return showers.sum(axis=(1, 2, 3))

def compute_total_event_hits(showers, threshold=0.0):
    """Total number of hits per event. Returns (K,)."""
    return (showers > threshold).sum(axis=(1, 2, 3))

def compute_cell_energy(showers):
    """All non-zero cell energies, flattened. Returns (M,)."""
    flat = showers.flatten()
    return flat[flat > 0]

def compute_cell_log_energy(showers):
    """log₁₀ of all non-zero cell energies. Returns (M,)."""
    return np.log10(compute_cell_energy(showers) + 1e-16)


# ─── Observable plot helpers ──────────────────────────────────────────────────

def _cmp_hist(ax_top, ax_bot, ref, gen, bins, xlabel, ylabel, gen_label, log_y=True):
    """Overlaid histogram + ratio panel. Returns EMD."""
    eps = 1e-8
    ref = np.asarray(ref).flatten()
    gen = np.asarray(gen).flatten()
    if isinstance(bins, int):
        lo = min(ref.min(), gen.min())
        hi = max(ref.max(), gen.max())
        bins = np.linspace(lo, hi, bins + 1)
    c_ref, edges, _ = ax_top.hist(ref, bins=bins, label='Geant4',
                                   color='silver', histtype='stepfilled')
    emd = wasserstein_distance(ref, gen)
    c_gen, _, _ = ax_top.hist(gen, bins=bins,
                               label=f'{gen_label} (EMD={emd:.3e})',
                               color='steelblue', histtype='step', linewidth=1.5)
    if log_y:
        ax_top.set_yscale('log')
    ax_top.set_ylabel(ylabel)
    ax_top.legend(fontsize=7)
    ax_top.tick_params(labelbottom=False)
    xc = 0.5 * (edges[:-1] + edges[1:])
    ratio = c_gen / (c_ref + eps)
    ax_bot.axhline(1., color='k', linestyle='--', linewidth=0.8)
    ax_bot.plot(xc, ratio, 'o', color='steelblue', markersize=2)
    ax_bot.set_ylim(0, 2)
    ax_bot.set_ylabel('Ratio')
    ax_bot.set_xlabel(xlabel)
    return emd


def _cmp_profile(ax_top, ax_bot, ref_prof, gen_prof, centers, xlabel, ylabel, gen_label):
    """Step-plot profile comparison + ratio panel. Returns EMD."""
    eps = 1e-8
    ref_prof = np.asarray(ref_prof, dtype=float)
    gen_prof = np.asarray(gen_prof, dtype=float)
    centers  = np.asarray(centers,  dtype=float)
    ref_w = ref_prof / (ref_prof.sum() + eps)
    gen_w = gen_prof / (gen_prof.sum() + eps)
    emd   = wasserstein_distance(centers, centers, ref_w, gen_w)
    step  = (centers[1] - centers[0]) if len(centers) > 1 else 1.
    edges = np.append(centers - step / 2, centers[-1] + step / 2)
    ax_top.fill_between(edges, np.append(ref_prof, ref_prof[-1]),
                        step='post', color='silver', alpha=1., label='Geant4')
    ax_top.step(edges, np.append(ref_prof, ref_prof[-1]),
                where='post', color='gray', linewidth=0.5)
    ax_top.step(edges, np.append(gen_prof, gen_prof[-1]),
                where='post', color='steelblue', linewidth=1.5,
                label=f'{gen_label} (EMD={emd:.3e})')
    ax_top.set_ylabel(ylabel)
    ax_top.legend(fontsize=7)
    ax_top.tick_params(labelbottom=False)
    ratio = gen_prof / (ref_prof + eps)
    ax_bot.axhline(1., color='k', linestyle='--', linewidth=0.8)
    ax_bot.plot(centers, ratio, 'o', color='steelblue', markersize=3)
    ax_bot.set_ylim(0, 2)
    ax_bot.set_ylabel('Ratio')
    ax_bot.set_xlabel(xlabel)
    return emd


def _obs_figure(n_cols, titles, suptitle='', figw=5, figh=6):
    """Figure with n_cols (top+ratio) panel pairs. Returns (fig, [(ax_top, ax_bot), ...])."""
    fig = plt.figure(figsize=(figw * n_cols, figh))
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    outer = gridspec.GridSpec(1, n_cols, figure=fig, wspace=0.4)
    axes = []
    for col, title in enumerate(titles):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col], height_ratios=[3, 1], hspace=0.05)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        ax_top.set_title(title, fontsize=9)
        axes.append((ax_top, ax_bot))
    return fig, axes


def _event_energy_grid(ref_ee, gen_ee, n_bins, suptitle, label_prefix, gen_label):
    """Grid of per-bin energy histograms. Returns fig."""
    n = ref_ee.shape[1]
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    fig.suptitle(suptitle, fontsize=12)
    axes_flat = np.array(axes).flatten()
    for i in range(n):
        rv, gv = ref_ee[:, i], gen_ee[:, i]
        hi = max(rv.max(), gv.max()) + 1e-3
        b  = np.linspace(0, hi, n_bins + 1)
        axes_flat[i].hist(rv, bins=b, color='silver', histtype='stepfilled', label='Geant4')
        axes_flat[i].hist(gv, bins=b, color='steelblue', histtype='step',
                          linewidth=1.5, label=gen_label)
        axes_flat[i].set_yscale('log')
        axes_flat[i].set_title(f'{label_prefix} {i}', fontsize=8)
        axes_flat[i].tick_params(labelsize=7)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=8)
    fig.tight_layout()
    return fig


def plot_long_observables(ref_showers, gen_showers, threshold=0.0, gen_label='Generated',
                           bins=50):
    """Plot all longitudinal observables. Returns (fig_profiles, fig_event_energy)."""
    Z = ref_showers.shape[3]
    z = np.arange(Z, dtype=float)

    fig, axes = _obs_figure(4, ['LongTotalEnergy', 'LongTotalHits',
                                  'LongFirstMoment', 'LongSecondMoment'],
                             suptitle='Longitudinal Profile Observables')
    _cmp_profile(*axes[0], compute_long_total_energy(ref_showers),
                 compute_long_total_energy(gen_showers), z, 'Z-layer', 'Mean Energy [MeV]', gen_label)
    _cmp_profile(*axes[1], compute_long_total_hits(ref_showers, threshold),
                 compute_long_total_hits(gen_showers, threshold), z, 'Z-layer', 'Mean Hits', gen_label)
    _cmp_hist(*axes[2], compute_long_first_moment(ref_showers),
              compute_long_first_moment(gen_showers), bins, '<z>', 'Events', gen_label)
    _cmp_hist(*axes[3], compute_long_second_moment(ref_showers),
              compute_long_second_moment(gen_showers), bins, '<z²>', 'Events', gen_label)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    n_show = min(9, Z)
    layer_idcs = np.linspace(0, Z - 1, n_show, dtype=int)
    ref_ee = compute_long_event_energy(ref_showers)[:, layer_idcs]
    gen_ee = compute_long_event_energy(gen_showers)[:, layer_idcs]
    fig_ev = _event_energy_grid(ref_ee, gen_ee, bins,
                                 'LongEventEnergy — Per-Z-layer Energy Distribution',
                                 'Z-layer', gen_label)
    for i, ax in enumerate(np.array(fig_ev.axes).flatten()[:n_show]):
        ax.set_title(f'Z-layer {layer_idcs[i]}', fontsize=8)

    return fig, fig_ev


def plot_rad_observables(ref_showers, gen_showers, threshold=0.0, gen_label='Generated',
                          bins=50):
    """Plot all radial observables. Returns (fig_profiles, fig_event_energy)."""
    R = ref_showers.shape[1]
    r = np.arange(R, dtype=float)

    fig, axes = _obs_figure(4, ['RadTotalEnergy', 'RadTotalHits',
                                  'RadFirstMoment', 'RadSecondMoment'],
                             suptitle='Radial Profile Observables')
    _cmp_profile(*axes[0], compute_rad_total_energy(ref_showers),
                 compute_rad_total_energy(gen_showers), r, 'R-bin', 'Mean Energy [MeV]', gen_label)
    _cmp_profile(*axes[1], compute_rad_total_hits(ref_showers, threshold),
                 compute_rad_total_hits(gen_showers, threshold), r, 'R-bin', 'Mean Hits', gen_label)
    _cmp_hist(*axes[2], compute_rad_first_moment(ref_showers),
              compute_rad_first_moment(gen_showers), bins, '<r>', 'Events', gen_label)
    _cmp_hist(*axes[3], compute_rad_second_moment(ref_showers),
              compute_rad_second_moment(gen_showers), bins, '<r²>', 'Events', gen_label)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_ev = _event_energy_grid(compute_rad_event_energy(ref_showers),
                                 compute_rad_event_energy(gen_showers),
                                 bins, 'RadEventEnergy — Per-R-bin Energy Distribution',
                                 'R-bin', gen_label)
    return fig, fig_ev


def plot_azim_observables(ref_showers, gen_showers, threshold=0.0, gen_label='Generated',
                           bins=50):
    """Plot all azimuthal observables. Returns (fig_profiles, fig_event_energy)."""
    P = ref_showers.shape[2]
    phi = np.arange(P, dtype=float)

    fig, axes = _obs_figure(4, ['AzimTotalEnergy', 'AzimTotalHits',
                                  'AzimFirstMoment', 'AzimSecondMoment'],
                             suptitle='Azimuthal Profile Observables')
    _cmp_profile(*axes[0], compute_azim_total_energy(ref_showers),
                 compute_azim_total_energy(gen_showers), phi, 'φ-bin', 'Mean Energy [MeV]', gen_label)
    _cmp_profile(*axes[1], compute_azim_total_hits(ref_showers, threshold),
                 compute_azim_total_hits(gen_showers, threshold), phi, 'φ-bin', 'Mean Hits', gen_label)
    _cmp_hist(*axes[2], compute_azim_first_moment(ref_showers),
              compute_azim_first_moment(gen_showers), bins, '<φ>', 'Events', gen_label)
    _cmp_hist(*axes[3], compute_azim_second_moment(ref_showers),
              compute_azim_second_moment(gen_showers), bins, '<φ²>', 'Events', gen_label)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig_ev = _event_energy_grid(compute_azim_event_energy(ref_showers),
                                 compute_azim_event_energy(gen_showers),
                                 bins, 'AzimEventEnergy — Per-φ-bin Energy Distribution',
                                 'φ-bin', gen_label)
    return fig, fig_ev


def plot_global_observables(ref_showers, gen_showers, threshold=0.0, gen_label='Generated',
                             bins=50):
    """Plot global shower observables (5 panels). Returns fig."""
    fig, axes = _obs_figure(5, ['TotalEventEnergy', 'TotalEventHits',
                                  'CellEnergy', 'CellLogEnergy', 'CellEnergy_xlog'],
                             suptitle='Global Shower Observables', figw=5, figh=6)

    ref_te = compute_total_event_energy(ref_showers)
    gen_te = compute_total_event_energy(gen_showers)
    _cmp_hist(*axes[0], ref_te, gen_te, bins, 'Total Energy [MeV]', 'Events', gen_label)

    ref_th = compute_total_event_hits(ref_showers, threshold)
    gen_th = compute_total_event_hits(gen_showers, threshold)
    _cmp_hist(*axes[1], ref_th, gen_th, bins, 'Total Hits', 'Events', gen_label, log_y=False)

    ref_ce = compute_cell_energy(ref_showers)
    gen_ce = compute_cell_energy(gen_showers)
    _cmp_hist(*axes[2], ref_ce, gen_ce, bins, 'Cell Energy [MeV]', 'Hits', gen_label)

    ref_le = compute_cell_log_energy(ref_showers)
    gen_le = compute_cell_log_energy(gen_showers)
    _cmp_hist(*axes[3], ref_le, gen_le, bins, 'log₁₀(Cell Energy)', 'Hits', gen_label)

    lo = max(ref_ce.min(), 1e-6)
    hi = ref_ce.max()
    bins_log = np.geomspace(lo, hi, bins + 1)
    _cmp_hist(*axes[4], ref_ce, gen_ce, bins_log, 'Cell Energy [MeV]', 'Hits', gen_label)
    axes[4][0].set_xscale('log')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_transverse_profile(gt, gen, cfg: dict, e_inc_gev=None):
    """Mean transverse profile (averaged over depth) comparison: GT vs generated.

    Parameters
    ----------
    gt, gen : ndarray (N, D, H, W)
    cfg : dict — must contain ``voxel_shape``.
    e_inc_gev : float or None
    """
    _, H, W     = cfg["voxel_shape"]
    gen_2d      = gen.sum(1).mean(0)   # (H, W)
    real_2d     = gt.sum(1).mean(0)    # (H, W)
    theta_edges = np.linspace(0, 2 * np.pi, H + 1)
    rho_edges   = np.arange(W + 1)
    vmin, vmax  = real_2d.min(), real_2d.max()

    title = 'Transverse Shower Profile: Generated vs. Real'
    if e_inc_gev is not None:
        title += f' (E_inc = {e_inc_gev:.0f} GeV)'

    fig, (ax_gen, ax_real) = plt.subplots(1, 2, figsize=(10, 5),
                                           subplot_kw={'projection': 'polar'})
    fig.suptitle(title, fontsize=14)

    for ax, data_2d, t in [
        (ax_gen,  gen_2d,  'Generated'),
        (ax_real, real_2d, 'Real (Test Set)'),
    ]:
        mesh = ax.pcolormesh(theta_edges, rho_edges, data_2d.T,
                             cmap='gist_grey', vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=11, pad=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    cbar = fig.colorbar(mesh, ax=[ax_gen, ax_real], orientation='vertical',
                        shrink=0.75, pad=0.01)
    cbar.set_label('Total Energy (summed over depth)')
    return fig, (ax_gen, ax_real)
