import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm
import skimage.util
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import imageio

# ========== CONFIG ==========
epoch = 29
method = "tSNE"
arg_name = "perplexity"
arg_value = 50
frame_times = 8
zoom = 0.7
invert_colors = False
n_videos = 500
subsample_videos = 1
patch_shape = (50, 50, 3)
patch_stride = [0.5, 0.5]
aggregation = False
aggregate_by = "frame"  # "frame" or "video"
aggregation_mode = "mean"
seed = 42

features_dir = f"/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_features/epoch_{epoch}/"
frame_root = "/data1/crops/random_walk_frames/IR_108_cm_years-2013/train_256"

if aggregation:
    save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_seed_{seed}_{n_videos}_videos_agg_{aggregate_by}_{aggregation_mode}"
else:
    save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_seed_{seed}_{n_videos}_videos"
csv_file = os.path.join(features_dir, f"{save_filename}.csv")
output_dir = os.path.join(features_dir, f"frames_embedding_{method}_frame_evolution")
os.makedirs(output_dir, exist_ok=True)

# ========== LOAD EMBEDDING DATA ==========
df = pd.read_csv(csv_file)
#assert all(col in df.columns for col in ['video_idx', 'tsne_x', 'tsne_y', 'node_idx'])

video_folders = sorted(os.listdir(frame_root))

def get_image_folder(row):
    try:
        folder = video_folders[int(row['video_idx'])]
        return os.path.join(frame_root, folder)
    except IndexError:
        print(f"Invalid video_idx {row['video_idx']}")
        return None

df['path'] = df.apply(get_image_folder, axis=1)

# ========== PATCH EXTRACTOR ==========
def patch_grid_no_aug(shape=(32, 32, 3), stride=[0.5, 0.5]):
    stride_px = [int(shape[0]*stride[0]), int(shape[1]*stride[1]), shape[2]]
    def extract_patches(img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        windows = skimage.util.view_as_windows(img, shape, step=stride_px)
        return windows.reshape(-1, *shape)
    return extract_patches

extract_patches = patch_grid_no_aug(patch_shape, patch_stride)

# ========== PLOT FUNCTION ==========
# def draw_colored_border(img, color, border_thickness=3):
#     """Draw a colored border around an image."""
#     bordered = Image.new("RGB", (img.width + 2*border_thickness, img.height + 2*border_thickness), color)
#     bordered.paste(img, (border_thickness, border_thickness))
#     return bordered

def plot_all_frames(df, out_path, frame_times, aggregation=None):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Normalize for layout
    x = df['tsne_x'].values
    y = df['tsne_y'].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    norm = mcolors.Normalize(vmin=0, vmax=frame_times - 1)
    cmap = cm.get_cmap("viridis", frame_times)

    for frame_idx in tqdm(range(frame_times), desc="Processing frames"):
        df_frame = df.copy()  # use all rows every time for uniform layout
        for i, row in df_frame.iterrows():
            img_path = os.path.join(row['path'], f"{frame_idx}.png")
            if not os.path.exists(img_path):
                continue
            try:
                full_img = Image.open(img_path).convert("RGB")
                if aggregation:
                    patch = full_img
                else:
                    patches = extract_patches(full_img)
                    node_idx = int(row['node_idx'])
                    if node_idx >= len(patches):
                        continue
                    patch = Image.fromarray(patches[node_idx])
                if invert_colors:
                    patch = ImageOps.invert(patch)

                # Add color-coded border based on timestamp
                #color_rgb = tuple((np.array(cmap(norm(frame_idx)))[:3] * 255).astype(np.uint8))
                #patch = draw_colored_border(patch, color=color_rgb, border_thickness=3)

                imbox = OffsetImage(patch, zoom=zoom)
                ab = AnnotationBbox(imbox, (x_norm[i], y_norm[i]), frameon=False)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()

    out_file = os.path.join(out_path, f"{save_filename}_all_frames.png")
    fig.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to: {out_file}")


def plot_frames_to_gif(
    df,
    plot_limits,
    out_path,
    frame_times,
    mode="dots",  # "dots" or "images"
    save_filename="tsne_viz",
    dot_size=50,
    zoom=0.7,
    aggregation=False,
    invert_colors=False
):
    os.makedirs(out_path, exist_ok=True)

    # Assign color by video_idx
    #video_indices = df['video_idx'].values
    unique_videos = sorted(df['video_idx'].unique())
    cmap = cm.get_cmap("tab10", len(unique_videos))
    video_colors = {vid: cmap(i) for i, vid in enumerate(unique_videos)}
    df['color'] = df['video_idx'].map(video_colors)
    #print(df)

    frame_images = []

    for frame_idx in tqdm(range(frame_times), desc="Generating frames"):
        fig, ax = plt.subplots(figsize=(8, 8))
        #filer onli the frame idx needed
        df_frame = df[df['frame_idx'] == frame_idx] 

        if mode == "dots":
            ax.scatter(
                df_frame['tsne_x'],
                df_frame['tsne_y'],
                c=df_frame['color'].tolist(),
                s=dot_size,
                edgecolors='k',
                linewidths=0.3
            )

        elif mode == "images":
            for i, row in df_frame.iterrows():
                img_path = os.path.join(row['path'], f"{frame_idx}.png")
                if not os.path.exists(img_path):
                    continue

                try:
                    full_img = Image.open(img_path).convert("RGB")
                    if aggregation:
                        patch = full_img
                    else:
                        patches = extract_patches(full_img)  # assumes this function exists
                        node_idx = int(row['node_idx'])
                        if node_idx >= len(patches):
                            continue
                        patch = Image.fromarray(patches[node_idx])

                    if invert_colors:
                        patch = ImageOps.invert(patch)

                    imbox = OffsetImage(patch, zoom=zoom)
                    ab = AnnotationBbox(imbox, (row['tsne_x'], row['tsne_y']), frameon=False)
                    ax.add_artist(ab)

                except Exception as e:
                    print(f"Skipping {img_path}: {e}")

        else:
            raise ValueError("Invalid mode: use 'dots' or 'images'")

        ax.set_xlim(plot_limits[0], plot_limits[1])
        ax.set_ylim(plot_limits[2], plot_limits[3])
        #ax.axis('off')
        plt.tight_layout()
        plt.title(f"Frame {frame_idx + 1}", fontsize=16, fontweight='bold')
        frame_path = os.path.join(out_path, f"{save_filename}_{frame_idx:03d}.png")
        fig.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()

        frame_images.append(Image.open(frame_path))

    # Save final GIF
    gif_path = os.path.join(out_path, f"{save_filename}_{len(unique_videos)}_videos_{mode}.gif")
    imageio.mimsave(gif_path, frame_images, duration=1000)
    print(f"Saved GIF to: {gif_path}")

# ========== RUN ==========
#plot_all_frames(df, output_dir, frame_times, aggregation)

#find random indeces to plot
print(df)
#find x and y limits
plot_limits = (df['tsne_x'].min(), df['tsne_x'].max() , df['tsne_y'].min() , df['tsne_y'].max())

unique_videos = sorted(df['video_idx'].unique())
random_indices = np.random.choice(unique_videos, size=min(subsample_videos, len(unique_videos)), replace=False)
df = df[df['video_idx'].isin(random_indices)]
print(df)
plot_frames_to_gif(df, plot_limits, output_dir,frame_times,mode='images')