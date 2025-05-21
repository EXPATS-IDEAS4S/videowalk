import os
import json
import numpy as np
import pandas as pd

# === SETTINGS ===
epoch = 29
aggregate_by = "video"  # "frame" or "video"
aggregation_mode = "mean"  # "mean", "sum", or "concat" TODO implement some other modes

# === LOAD FEATURES AND METADATA ===
features_dir = f"/data1/fig/videowalk/run-1_30-epochs_8k-clips/extracted_features/epoch_{epoch}/"
features_file = os.path.join(features_dir, "features.npy")
metadata_file = os.path.join(features_dir, "features_metadata.json")

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
df_full = pd.concat([df_meta, features_df], axis=1)

# === GROUP AND AGGREGATE FEATURES ===
if aggregate_by == "frame":
    group_keys = ["video_idx", "frame_idx"]
    print("Aggregating features per frame.")
elif aggregate_by == "video":
    group_keys = ["video_idx"]
    print("Aggregating features per video.")
else:
    raise ValueError("aggregate_by must be either 'frame' or 'video'")

if aggregation_mode == "mean":
    df_agg = df_full.groupby(group_keys).mean().reset_index()
elif aggregation_mode == "sum":
    df_agg = df_full.groupby(group_keys).sum().reset_index()
elif aggregation_mode == "concat":
    # Concatenate features into one long vector per group
    df_agg = df_full.groupby(group_keys).agg(
        lambda x: np.concatenate(x.values) if x.name.startswith("feature_") else x.iloc[0]
    ).reset_index()
    # Expand concatenated features to new columns
    feature_cols = [col for col in df_agg.columns if col.startswith("feature_")]
    df_agg = df_agg.drop(columns=feature_cols)
    df_agg[["feature_" + str(i) for i in range(df_agg.iloc[0][-1].shape[0])]] = pd.DataFrame(df_agg.iloc[:, -1].to_list())
    df_agg = df_agg.drop(columns=df_agg.columns[-1])  # remove the nested feature vector column
else:
    raise ValueError("aggregation_mode must be 'mean', 'sum', or 'concat'")

print(f"Aggregated DataFrame shape: {df_agg.shape}")

# === SAVE OUTPUT ===
output_file = os.path.join(features_dir, f"features_{aggregate_by}_{aggregation_mode}.npy")
np.save(output_file, df_agg.filter(like="feature_").values)

# Save updated metadata (video/frame indices)
meta_outfile = os.path.join(features_dir, f"features_{aggregate_by}_{aggregation_mode}_metadata.json")
df_agg[group_keys].to_json(meta_outfile, orient="records", indent=2)

print(f"Saved aggregated features to {output_file}")
print(f"Saved metadata to {meta_outfile}")
