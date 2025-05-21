import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import imageio
from tqdm import tqdm
from PIL import ImageOps

# ========== USER CONFIG ==========
epoch = 29
subsample_videos = 500  # number of unique videos to show per frame
method = "tSNE"
arg_name = "perplexity"
arg_value =  50
aggregation = True
aggregate_by = "video"  # "frame" or "video"
aggregation_mode = "mean"
grid_size = 15
zoom = 0.4
save_gif = True
frame_times = 8

# =================================
features_dir = f"/data1/fig/videowalk/run-1_30-epochs_8k-clips/extracted_features/epoch_{epoch}/"
frame_root = "/data1/crops/random_walk_frames/IR_108/train_256"

if aggregation:
    save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_{subsample_videos}_videos_agg_{aggregate_by}_{aggregation_mode}"
else:
    save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_{subsample_videos}_videos"

csv_file = os.path.join(features_dir, f"{save_filename}.csv")
output_dir = os.path.join(features_dir, f"frames_embedding_{method}")
os.makedirs(output_dir, exist_ok=True)

# === LOAD EMBEDDING DATA ===
df = pd.read_csv(csv_file, usecols=['video_idx', 'tsne_x', 'tsne_y'])
print(df)
print(f"Loaded CSV: {csv_file}, shape: {df.shape}")

# === ATTACH FRAME IMAGE PATHS TO DATAFRAME ===
# Step 1: Get sorted list of folder names (these are the real video names)
video_folders = sorted(os.listdir(frame_root))
print(f"Found {len(video_folders)} video folders.")

# Step 2: Map video_idx to actual folder name
def get_image_path(row):
    try:
        video_folder_name = video_folders[int(row['video_idx'])]  # real name from sorted list
    except IndexError:
        print(f"Invalid video_idx {row['video_idx']} (out of range)")
        return None
    #frame_file = f"{int(row['frame_idx'])}.png"
    return os.path.join(frame_root, video_folder_name)#, frame_file)

df['path'] = df.apply(get_image_path, axis=1)

# # === Ensure necessary columns exist ===
# required_cols = ['video_idx', 'frame_idx', 'path', 'Component_1', 'Component_2']
# for col in required_cols:
#     assert col in df.columns, f"Column '{col}' is required in the CSV."

# # === Subsample Videos ===
# unique_videos = df['video_idx'].unique()
# selected_videos = np.random.choice(unique_videos, subsample_videos, replace=False)
# df = df[df['video_idx'].isin(selected_videos)]

# # === Unique frame timestamps (sorted) ===
# frame_times = sorted(df['frame_idx'].unique())
# print(f"Plotting {len(frame_times)} frame timestamps.")

# === Binned Grid Plot Function ===
def plot_embedding_frame(df_frame, out_path, frame_idx):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = df_frame['tsne_x'].values
    y = df_frame['tsne_y'].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    ix = np.minimum((x_norm * grid_size).astype(int), grid_size - 1)
    iy = np.minimum((y_norm * grid_size).astype(int), grid_size - 1)
    centers = np.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size)

    placed = {}
    for i, j, xn, yn in zip(ix, iy, x_norm, y_norm):
        bin_key = (i, j)
        cx, cy = centers[i], centers[j]
        d = (xn - cx)**2 + (yn - cy)**2
        if bin_key not in placed or d < placed[bin_key][0]:
            placed[bin_key] = (d, df_frame.index[(ix == i) & (iy == j)][0])

    for (i, j), (_, idx) in placed.items():
        row = df_frame.loc[idx]
        try:
            img = Image.open(row['path']+'/'+str(frame_idx)+'.png').convert("RGB")
            img = ImageOps.invert(img)
            imbox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imbox, (centers[i], centers[j]), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Warning: Skipping image at {row['path']} due to error: {e}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()
    
    frame_filename = os.path.join(out_path, f"frame_{frame_idx:03d}.png")
    fig.savefig(frame_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return frame_filename

# === Generate per-frame plots ===
frame_image_paths = []
for t in tqdm(range(frame_times), desc="Generating plots"):
    #df_frame = df[df['frame_idx'] == t]
    #if df_frame.empty:
    #    continue
    img_path = plot_embedding_frame(df, output_dir, t)
    frame_image_paths.append(img_path)



# === Make GIF ===
if save_gif and frame_image_paths:
    gif_path = os.path.join(output_dir, f"embedding_evolution_{method}.gif")
    
    # Load all frames
    frames = [Image.open(fname).convert("RGB") for fname in frame_image_paths]

    # Save as GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000,  # 2000 ms = 2 seconds per frame
        loop=0  # loop forever
    )

    print(f"GIF saved to {gif_path}")
