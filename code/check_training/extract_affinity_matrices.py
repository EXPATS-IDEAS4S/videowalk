import torch
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tqdm import tqdm
import random
import sys
import torch.nn.functional as F

sys.path.append('/home/Daniele/codes/videowalk/code/')

from model import CRW
from data.video import VideoList
import utils

# ------------------ Load Config ------------------
config_path = '/home/Daniele/codes/videowalk/code/configs/config_test.yaml'
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

args = SimpleNamespace(**config_dict)

args.resume = f'/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/checkpoints/model_{args.epoch}.pth'
args.output_dir = f'/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_affinities/epoch_{args.epoch}'
os.makedirs(args.output_dir, exist_ok=True)

# ------------------ Load Model ------------------
class Args:
    pass

model = CRW(args).to(args.device)
checkpoint = torch.load(args.resume, map_location=args.device)
utils.partial_load(checkpoint['model'], model, skip_keys=['head'])
model.eval()

# ------------------ Load Dataset ------------------
transform = utils.augs.get_train_transforms(args)
video_dataset = VideoList(
    filelist=args.data_path,
    clip_len=args.clip_len,
    is_train=False,
    frame_gap=args.frame_skip,
    transform=transform,
    random_clip=False
)

dataloader = torch.utils.data.DataLoader(
    video_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.workers // 2,
    pin_memory=True
)

# ------------------ Compute Patch-wise Affinity ------------------
selected_idxs = random.sample(range(len(video_dataset)), k=10)
all_affinity_vectors = []  # Shape: [num_videos, N*N*(T-1)]
temperature = args.temp


# Plotting function
def plot_affinity_timeline(patch_affinities, video_idx, output_dir):
    """
    Plots a row of heatmaps for affinity matrices between consecutive frames.
    Each heatmap corresponds to frame t -> t+1 for the given video.
    """
    num_steps = len(patch_affinities)
    fig, axs = plt.subplots(1, num_steps, figsize=(2.5 * num_steps, 3), constrained_layout=True)

    # If only one matrix, axs is not a list
    if num_steps == 1:
        axs = [axs]

    vmin, vmax = 0, 1  # Normalize color scale across all matrices

    for i, A in enumerate(patch_affinities):
        #print(A.shape)
        ax = axs[i]
        im = ax.imshow(A, cmap='viridis', vmin=vmin, vmax=vmax)

        # Set major ticks at each cell boundary
        ax.set_xticks(np.arange(-0.5, A.shape[1], 1), minor=False)
        ax.set_yticks(np.arange(-0.5, A.shape[0], 1), minor=False)

        # Add gridlines based on the ticks
        ax.grid(which='major', color='w', linestyle='-', linewidth=0.5)

        ax.set_title(f"t{i}→t{i+1}")
        ax.set_yticks([])
        ax.set_xticks([])

        # Only add colorbar to the last subplot
        if i == num_steps - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

    save_path = os.path.join(output_dir, f"affinity_timeline_{video_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

with torch.no_grad():
    for video_idx, (imgs, imgs_orig, meta) in enumerate(tqdm(dataloader)):
        imgs = imgs[0]  # Since it's a list of 1 tensor: (T, C*N, H, W)
        #print(imgs.shape)    

        B, T, cn, H, W = imgs.shape
        C = 3
        N = cn // C
        assert cn % C == 0, "Channel dimension not divisible by C=3"

        # Reshape to (B, T, N, C, H, W), then permute to (B, N, C, T, H, W)
        imgs = imgs.view(B, T, N, C, H, W)       # (B, T, N, C, H, W)
        imgs = imgs.permute(0, 2, 3, 1, 4, 5)    # (B, N, C, T, H, W)
        #print(imgs.shape)

        feats, maps = model.pixels_to_nodes(imgs.to(args.device))  # [1, C, T, N]
        feats = feats.squeeze(0)  # [C, T, N]
        #print(feats.shape)
        

        patch_affinities = []
        for t in range(T - 1):
            f1 = feats[:, t, :]       # [C, N]
            f2 = feats[:, t + 1, :]   # [C, N]

            # Add batch and time dimensions: [1, C, 1, N]
            f1 = f1.unsqueeze(0).unsqueeze(2)  # [1, C, 1, N]
            f2 = f2.unsqueeze(0).unsqueeze(2)  # [1, C, 1, N]

            # Compute affinity: [1, 1, N, N]
            A = torch.einsum('bctn,bctm->btnm', f1, f2)
            #print(A.shape)

            # Remove batch/time dims: [N, N]
            A = A.squeeze(0).squeeze(0)
            
            # Apply temperature-scaled softmax to get stochastic matrix
            A = F.softmax(A / temperature, dim=-1)  # row-wise softmax (probabilities)

            patch_affinities.append(A.cpu().numpy())

        # Save long vector for all time steps (flatten N×N×(T-1))
        A_concat = np.stack(patch_affinities)  # [T-1, N, N]
        #print(A_concat.shape)

        #print(A_concat.shape)
        A_flat = A_concat.reshape(-1)  # [N * N * (T-1)]
        #print(A_flat.shape)
        all_affinity_vectors.append(A_flat)

        #np.save(os.path.join(args.output_dir, f"affinity_vector_{i}.npy"), A_flat)

        # Plot the whole affinity row for selected videos
        if video_idx in selected_idxs:
            plot_affinity_timeline(patch_affinities, video_idx, args.output_dir)

# Save full array: [n_videos, N*N*(T-1)]
all_affinity_vectors = np.stack(all_affinity_vectors)
np.save(os.path.join(args.output_dir, "all_affinity_vectors.npy"), all_affinity_vectors)

