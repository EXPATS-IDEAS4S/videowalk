import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
from PIL import Image
import matplotlib.patches as patches

def load_affinity_vectors(path):
    return np.load(path)  # shape [num_videos, (T-1)*N*N]

def get_video_dirs(base_dir, n=5):
    """
    Returns a list of n randomly selected video directories from base_dir,
    along with their indices in the full sorted list.
    
    Args:
        base_dir (str): The root directory containing video folders.
        n (int): Number of video directories to randomly sample.
    
    Returns:
        selected_dirs (list of str): Randomly selected video directory paths.
        selected_indices (list of int): Indices of selected_dirs in the full sorted list.
    """
    all_dirs = sorted([d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)])
    
    selected_indices = random.sample(range(len(all_dirs)), n)
    selected_dirs = [all_dirs[i] for i in selected_indices]
    
    return selected_dirs, selected_indices

def load_video_frames(video_dir, max_frames=8):
    frame_paths = sorted(glob.glob(os.path.join(video_dir, '*.png')))[:max_frames]
    return [Image.open(p).convert('RGB') for p in frame_paths]

def plot_video_with_patch_paths(frames, affinity_vector, out_path, patch_size=(50, 50), stride=(25, 25)):
    T = len(frames)
    img_h, img_w = frames[0].height, frames[0].width
    ph, pw = patch_size
    sh, sw = stride

    # Compute patch top-left positions
    patch_coords = []
    for y in range(0, img_h - ph + 1, sh):
        for x in range(0, img_w - pw + 1, sw):
            patch_coords.append((x, y))
    N = len(patch_coords)

    # Reshape affinity vector to (T-1, N, N)
    A = affinity_vector.reshape((T - 1, N, N))

    fig, axs = plt.subplots(N, T, figsize=(T * 3, N * 3))

    for patch_start in range(N):
        # Trace most probable path from each patch
        path = [patch_start]
        for t in range(T - 1):
            next_patch = A[t, path[-1]].argmax()
            path.append(next_patch)

        for t in range(T):
            ax = axs[patch_start, t] if N > 1 else axs[t]
            ax.imshow(frames[t])
            ax.axis('off')
            if t == 0:
                ax.set_ylabel(f'Patch {patch_start}', fontsize=10)

            # Get top-left corner of patch
            x, y = patch_coords[path[t]]
            rect = patches.Rectangle(
                (x, y),
                pw,
                ph,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

epoch = 29
run_name = "3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel"
aff_path = f"/data1/runs/videowalk/{run_name}/extracted_affinities/epoch_{epoch}/all_affinity_vectors.npy"
video_root = "/data1/crops/random_walk_frames/IR_108_cm_years-2013/train_256/"
output_dir = f"/data1/runs/videowalk/{run_name}/extracted_affinities/epoch_{epoch}/affinity_path_viz"
os.makedirs(output_dir, exist_ok=True)

affinities = load_affinity_vectors(aff_path)  # Shape: (num_videos, vector_dim)
print("Affinity matrix shape:", affinities.shape)

# Get video dirs and corresponding indices
video_dirs, indices = get_video_dirs(video_root, n=5)
print(len(video_dirs), len(indices))
print("Selected video directories:")

for i, (vid_dir, idx) in enumerate(zip(video_dirs, indices)):
    print(f"\nProcessing video {i+1}/{len(video_dirs)}: {vid_dir}")
    
    affinity_vector = affinities[idx]  # Get correct affinity row
    print("Affinity vector shape:", affinity_vector.shape)

    frames = load_video_frames(vid_dir, max_frames=8)  # Customize max_frames if needed
    out_path = os.path.join(output_dir, f"video_{idx}_paths.png")
    plot_video_with_patch_paths(frames, affinity_vector, out_path)

