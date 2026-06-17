import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wasserstein_distance
import torch
from torch.utils.data import Dataset


# ─── I/O ──────────────────────────────────────────────────────────────────────

class LEMURS(Dataset):
    """Raw calorimeter-shower dataset — no preprocessing.

    Loads and reshapes voxels; optionally applies a model-specific transform.

    Parameters
    ----------
    transform : callable(shower_np, energy_mev) -> (x_tensor, cond_tensor), optional
        Applied in __getitem__.  When None, returns raw
        ``(Tensor(D,H,W) float32, Tensor() float32 in MeV)``.

    Attributes
    ----------
    showers  : ndarray (N, D, H, W) float32  — raw MeV deposits
    energies : ndarray (N,) float32          — incident energy in MeV
    """

    def __init__(self, path: str, showers_key: str, energies_key: str,
                 voxel_shape: tuple, data_slice=None, transform=None):
        super().__init__()
        self.voxel_shape = voxel_shape
        self.transform   = transform

        with h5py.File(path, "r") as f:
            raw_showers  = f[showers_key][data_slice if data_slice is not None else ()]
            raw_energies = f[energies_key][data_slice if data_slice is not None else ()]

        self.showers  = raw_showers.astype(np.float32).reshape(-1, *voxel_shape)
        self.energies = raw_energies.astype(np.float32).flatten()

    @classmethod
    def from_config(cls, cfg: dict, train_split: float = 0.8, transform=None):
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
        """
        with h5py.File(cfg["data_path"], "r") as f:
            total = len(f[cfg["showers_key"]]) if cfg.get("n_samples") is None else cfg["n_samples"]

        train_size = int(train_split * total)
        train_ds = cls(cfg["data_path"], cfg["showers_key"], cfg["energies_key"],
                       cfg["voxel_shape"], data_slice=slice(0, train_size),
                       transform=transform)
        test_ds  = cls(cfg["data_path"], cfg["showers_key"], cfg["energies_key"],
                       cfg["voxel_shape"], data_slice=slice(train_size, total),
                       transform=transform)

        print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Voxel shape: {cfg['voxel_shape']}")
        return train_ds, test_ds

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, idx):
        shower = self.showers[idx]       # (D, H, W)
        energy = self.energies[idx]      # scalar MeV
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
        with h5py.File(cfg["data_path"], "r") as f:
            total = len(f["showers"]) if cfg.get("n_samples") is None else cfg["n_samples"]

        idcs       = np.arange(total)
        train_size = int(train_split * total)
        train_ds   = cls(cfg["data_path"], hlf, i_idcs=idcs[:train_size], transform=transform)
        test_ds    = cls(cfg["data_path"], hlf, i_idcs=idcs[train_size:], transform=transform)

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


def plot_z_profiles(showers, filename='', i_idcs=None):
    """Plot linear and log z-profiles (summed over R and PHI axes) for n_samples showers.

    Parameters
    ----------
    showers : ndarray (K, R, PHI, Z)
    filename : str
        Used in the figure title.
    i_idcs : array-like or None
        Slice descriptor used in the figure title.
    n_samples : int or None
        Number of showers to plot; None plots all.
    """
    z_profiles = showers.sum(axis=(1, 2))
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

    fig   = plt.figure(figsize=(6 * n_cols, 10))
    outer = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.5, wspace=0.35)

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

    plt.suptitle(f'Shower Comparison: Geant4 vs {gen_label}', fontsize=14, y=1.01)
    return fig
