import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd


def compute_tsne(input_npy_path, output_npy_path, perplexity=30, random_state=42):
    """
    Loads affinity vectors, applies t-SNE, and saves reduced features to a .npy file.
    
    Args:
        input_npy_path (str): Path to the input .npy file (shape: [N, D]).
        output_npy_path (str): Path to save the reduced 2D features (shape: [N, 2]).
        perplexity (int): t-SNE perplexity parameter.
        random_state (int): Random seed.
    """

    X = np.load(input_npy_path)
    print(f"[t-SNE] Loaded input with shape: {X.shape}")

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca')
    X_reduced = tsne.fit_transform(X)
    print(f"[t-SNE] Reduced shape: {X_reduced.shape}")

    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, X_reduced)
    print(f"[t-SNE] Saved reduced features to: {output_npy_path}")


def save_csv_with_paths(tsne_npy_path, dir_vid, output_csv_path):
    """
    Combines 2D t-SNE features with corresponding video folder paths and saves as a CSV.
    
    Args:
        tsne_npy_path (str): Path to .npy file containing 2D features (shape: [N, 2]).
        dir_vid (str): Directory containing one subfolder per video, in order.
        output_csv_path (str): Path to save the output CSV.
    """
    # Load t-SNE data
    X_reduced = np.load(tsne_npy_path)
    print(f"[CSV] Loaded reduced features with shape: {X_reduced.shape}")

    # List subfolders (e.g., one per video)
    subfolders = sorted([f.path for f in os.scandir(dir_vid) if f.is_dir()])
    print(f"[CSV] Found {len(subfolders)} subfolders in: {dir_vid}")

    if len(subfolders) != X_reduced.shape[0]:
        raise ValueError(f"Mismatch: {len(subfolders)} subfolders vs {X_reduced.shape[0]} feature rows")

    # Create DataFrame
    df = pd.DataFrame(X_reduced, columns=["component_1", "component_2"])
    df["video_path"] = subfolders

    # Save CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"[CSV] Saved CSV to: {output_csv_path}")



def plot_tsne(tsne_npy_path, output_plot_path, title="t-SNE of Affinity Vectors"):
    """
    Loads 2D t-SNE features and plots a scatter plot.
    
    Args:
        tsne_npy_path (str): Path to the saved 2D features .npy file.
        output_plot_path (str): Path to save the scatter plot PNG.
        title (str): Plot title.
    """
    X_reduced = np.load(tsne_npy_path)
    print(f"[Plot] Loaded reduced features with shape: {X_reduced.shape}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=10, alpha=0.7, cmap='viridis')
    plt.title(title)
    plt.axis("off")

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(output_plot_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved plot to: {output_plot_path}")



# Example usage
if __name__ == "__main__":
    # File paths
    epoch = 29
    main_dir =  f"/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_affinities/epoch_{epoch}/"
    input_npy = f"{main_dir}all_affinity_vectors.npy"
    perplexity = 30
    reduced_npy = f"{main_dir}tsne_affinities_perp-{perplexity}.npy"
    plot_path = f"{main_dir}tsne_perp-{perplexity}_plot.png"
    dir_vid = "/data1/crops/random_walk_frames/IR_108_cm_years-2013/train_256/"
    output_csv_path = f"{main_dir}tsne_affinities_with_paths_perp-{perplexity}.csv"
    n_frames = 8
    grid_size = 15
    zoom = 0.45
    from_selected_indices = True  # Set to True if you want to use selected indices from a previous run
    n_seed = 42
    subsample_videos = 300

    # Run steps
    compute_tsne(input_npy, reduced_npy, dir_vid, perplexity)
    save_csv_with_paths(reduced_npy, dir_vid, output_csv_path)
    plot_tsne(reduced_npy, plot_path)

    