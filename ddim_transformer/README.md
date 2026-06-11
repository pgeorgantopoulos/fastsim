# DDIM Training on LEMURS 3D Calorimeter Data

Denoising Diffusion Implicit Models (Song et al., 2020, https://arxiv.org/pdf/2010.02502) applied to 3D voxelized
calorimeter shower data from the CERN LEMURS dataset.

Training objective is identical to DDPM (predict ε), but sampling is non-Markovian,
allowing orders-of-magnitude fewer inference steps.
**CFG dict** — single config object covering data path, voxel geometry (45, 16, 9) matching LEMURS (N_layers, N_alpha, N_r), model width, diffusion schedule, and training hyperparameters. The user had already filled in the actual HDF5 path.

## LemursDataset

* reads the HDF5 file, reshapes the flat voxel array into (N, D, H, W)
* applies log-normalisation (log(E + ε)) to handle the sparse, heavy-tailed energy distribution, then scales to [-1, 1]. Returns (x, cond) where cond = log10(E_inc). (as in <https://arxiv.org/pdf/2308.03876>, Eq.8)


## UNet3D+Transformer — 3-D AutoEncoder:

* Sinusoidal time embedding fed through an MLP
* Incident-energy conditioned via a separate projection head
* FiLM-style (scale + shift) conditioning in every ResBlock3D
* SelfAttn3D at the coarsest spatial resolution
* Symmetric decoder with skip connections and F.interpolate to handle odd spatial dims

To integrate transformer blocks into the UNet3D, you would typically follow a strategy similar to what's done in Vision Transformers (ViT) or TransUNet (<https://arxiv.org/pdf/2102.04306>). The core idea is to replace or augment certain convolutional parts of the UNet with transformer layers, allowing the model to capture long-range dependencies more effectively.

Here are the main components and conceptual alterations needed:

1.  **Patch Embedding**: Instead of raw voxel data, the input to the transformer blocks needs to be a sequence of tokens. This is achieved by dividing the 3D voxel input into non-overlapping or overlapping 3D patches and then linearly projecting each patch into a higher-dimensional embedding space. This effectively flattens spatial information into a sequence.

2.  **Positional Encoding**: Since transformers are permutation-invariant (they don't inherently understand spatial relationships), you need to add positional embeddings to the patch embeddings. This injects spatial awareness back into the tokens.

3.  **Transformer Blocks**: These blocks consist of Multi-Head Self-Attention (MHSA) and a Feed-Forward Network (FFN), usually with Layer Normalization and residual connections. These blocks would operate on the sequence of patch embeddings.

4.  **Integration into UNet Architecture (Encoder/Decoder)**:
    *   **Encoder**: You could replace some of the downsampling convolutional blocks with sequences of patch embeddings followed by transformer blocks. The patch embedding effectively handles the initial 'tokenization' and potentially downsampling. You'd need a mechanism to progressively reduce the sequence length or token dimensions.
    *   **Bottleneck**: This is a common place to introduce a full-fledged transformer encoder, where the most abstract features are processed globally.
    *   **Decoder**: For the decoder, you'd need a mechanism to convert the processed tokens back into a spatial grid, perhaps using a 'patch expanding' or 'patch upsampling' layer (similar to inverse patch embedding or simple reshaping + convolutional layers) and then merge them with skip connections from the encoder. Skip connections would also need to be adapted to handle either token sequences or feature maps.

5.  **Conditioning**: The existing time and energy embeddings (`emb`) would need to be incorporated into the transformer blocks. This could be done by adding them to the patch embeddings before the transformer layers, or by using techniques like FiLM conditioning within the FFN of the transformer blocks, similar to how it's currently used in your ResBlock3D.

**DDIMScheduler** — cosine or linear β schedule, q_sample for the forward pass, training_loss (plain MSE on noise), and ddim_sample with configurable steps and η (0 = deterministic, 1 = DDPM-equivalent).

train() — standard loop with gradient clipping, cosine-annealing LR, and checkpoint save/resume logic.

Two things to adjust before running:

1. Verify voxel_shape matches your actual HDF5 file dimensions (check with h5py inspect)
2. Reduce ch_mults to (1, 2, 4) and base_ch to 16 if GPU memory is tight

# Notes/Issues

1 epoch for

    device: cuda
    Train Dataset: 80000 events | Test Dataset: 20000 events | Voxel shape: (45, 16, 9)
    Parameters : 7.92 M
    VRAM       : 0.029 GiB

**needs ~5 min / 10 ep**