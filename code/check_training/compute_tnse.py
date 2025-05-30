import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#import umap

# === CONFIGURATION ===
n_seed = 42
random.seed(n_seed)
np.random.seed(n_seed)

epoch = 29
features_dir = f"/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_features/epoch_{epoch}/"

subsample_num = 500  # number of unique video clips to highlight
method = "tSNE"  # method to use for dimensionality reduction (tSNE, UMAP)

aggregation = True
aggregate_by = "video"  # "frame" or "video"
aggregation_mode = "mean"  # "mean", "sum", or "concat"

if aggregation:
    features_file = os.path.join(features_dir, f"features_{aggregate_by}_{aggregation_mode}.npy")
    metadata_file = os.path.join(features_dir, f"features_{aggregate_by}_{aggregation_mode}_metadata.json")
else:
    features_file = os.path.join(features_dir, "features.npy")
    metadata_file = os.path.join(features_dir, "features_metadata.json")

# === LOAD FEATURES AND METADATA ===
if not os.path.exists(features_file):
    raise FileNotFoundError(f"Feature file not found: {features_file}")
features = np.load(features_file)
print(f"Loaded features with shape: {features.shape}")

if not os.path.exists(metadata_file):
    raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
with open(metadata_file, "r") as f:
    metadata = json.load(f)
print(f"Loaded metadata with {len(metadata)} rows.")
print(f"Metadata columns: {list(metadata[0].keys())}")

# === COMBINE FEATURES AND METADATA INTO DATAFRAME ===
df_meta = pd.DataFrame(metadata)
features_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
df_meta = pd.concat([df_meta, features_df], axis=1)
print(f"Merged DataFrame shape: {df_meta.shape}")

# === SELECT RANDOM VIDEO CLIPS ===
unique_videos = df_meta["video_idx"].unique()
selected_videos = random.sample(list(unique_videos), min(subsample_num, len(unique_videos)))
df_selected = df_meta[df_meta["video_idx"].isin(selected_videos)]
print(f"Selected {len(selected_videos)} unique videos, {df_selected.shape[0]} total rows.")

# === PREPARE SELECTED FEATURES FOR t-SNE ===
feature_cols = [f"feature_{i}" for i in range(features.shape[1])]
selected_feats = df_selected[feature_cols].values

print(selected_feats.shape)


# === RUN t-SNE ===

# # normalize the features before t-SNE
# scaler = StandardScaler()
# selected_feats = scaler.fit_transform(selected_feats)

# # Apply PCA to reduce dimensionality before t-SNE
# pca = PCA(n_components=50, random_state=42)
# selected_feats = pca.fit_transform(selected_feats)

print(f"Running {method}...")
if method == "tSNE":
    arg_name = "perplexity"
    arg_value = 50
    method_obj = TSNE(n_components=2, perplexity=arg_value, init='pca', random_state=n_seed)
elif method == "UMAP":
    arg_name = "n_neighbors"
    arg_value = 15
    #method_obj = umap.UMAP(n_components=2, n_neighbors=arg_value, random_state=42)
else:
    raise ValueError(f"Unknown method: {method}")

reduced = method_obj.fit_transform(selected_feats)

# === ADD t-SNE RESULTS BACK TO DATAFRAME ===
df_selected["tsne_x"] = reduced[:, 0]
df_selected["tsne_y"] = reduced[:, 1]

# === SAVE t-SNE RESULTS as csv
if aggregation:
    save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_{subsample_num}_videos_agg_{aggregate_by}_{aggregation_mode}"
else:
    save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_{subsample_num}_videos"

output_csv = os.path.join(features_dir, f"{save_filename}.csv")
df_selected.to_csv(output_csv, index=False)
print(f"t-SNE results saved to {output_csv}")

# === PLOT ===
plt.figure(figsize=(10, 10))

# Plot all points in gray
#plt.scatter(df_selected["tsne_x"], df_selected["tsne_y"], s=1, c="lightblue", alpha=0.5)

# Plot selected videos with colors
# if aggregation:
#     plt.scatter(df_selected["tsne_x"], df_selected["tsne_y"], s=5, alpha=0.7, c="blue")
# else:
if len(selected_videos)>1:
    for i, video_id in enumerate(selected_videos):
        clip_data = df_selected[df_selected["video_idx"] == video_id]
        plt.scatter(clip_data["tsne_x"], clip_data["tsne_y"], s=5, alpha=0.8)#label=f"Clip {video_id}")
else:
    clip_data = df_selected[df_selected["video_idx"] == selected_videos[0]]
    #find number of unique
    frames = clip_data["frame_idx"].unique()
    print(f"unique frames: {frames}")
    # Loop over different frame index
    for j, frame_id in enumerate(frames):
        frame_data = clip_data[clip_data["frame_idx"] == frame_id]
        plt.scatter(frame_data["tsne_x"], frame_data["tsne_y"], s=5, alpha=0.5)

plt.title(f"{method} ({arg_name} {arg_value}) of {subsample_num} Videos (Epoch {epoch})", fontsize=14, fontweight="bold")
plt.xlabel("t-SNE 1", fontsize=12)
plt.ylabel("t-SNE 2", fontsize=12)
#ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.legend(loc="best", markerscale=2)
plt.savefig(os.path.join(features_dir, f"{save_filename}.png"), dpi=300, bbox_inches='tight')
