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
n_videos = 3000  # number of unique videos to show per frame
subsample_videos = 300 # number of unique videos to subsample for visualization
method = "tSNE"
arg_name = "perplexity"
arg_value =  50
aggregation = True
aggregate_by = "video"  # "frame" or "video"
aggregation_mode = "mean"
grid_size = 15
zoom = 0.45
save_gif = True
frame_times = 8
n_seed = 42  # for reproducibility

#Define functions
def get_image_path(row):
    try:
        video_folder_name = video_folders[int(row['video_idx'])]  # real name from sorted list
    except IndexError:
        print(f"Invalid video_idx {row['video_idx']} (out of range)")
        return None
    #frame_file = f"{int(row['frame_idx'])}.png"
    return os.path.join(frame_root, video_folder_name)#, frame_file)


def normalize_tsne(df):
    x = df['tsne_x'].values
    y = df['tsne_y'].values
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    df = df.copy()
    df['x_norm'] = x_norm
    df['y_norm'] = y_norm
    return df


def place_images_on_grid(df, frame_idx, zoom, grid_size):
    x_norm = df['x_norm'].values
    y_norm = df['y_norm'].values
    ix = np.minimum((x_norm * grid_size).astype(int), grid_size - 1)
    iy = np.minimum((y_norm * grid_size).astype(int), grid_size - 1)
    centers = np.linspace(0.5 / grid_size, 1 - 0.5 / grid_size, grid_size)

    placed = {}
    for i, j, xn, yn in zip(ix, iy, x_norm, y_norm):
        bin_key = (i, j)
        cx, cy = centers[i], centers[j]
        d = (xn - cx) ** 2 + (yn - cy) ** 2
        if bin_key not in placed or d < placed[bin_key][0]:
            placed[bin_key] = (d, df.index[(ix == i) & (iy == j)][0])

    image_boxes = []
    used_video_indices = []

    for (i, j), (_, idx) in placed.items():
        row = df.loc[idx]
        img_box = load_and_box_image(row['path'], frame_idx, centers[i], centers[j], zoom)
        if img_box:
            image_boxes.append(img_box)
            used_video_indices.append(row['video_idx'])

    return image_boxes, used_video_indices


def place_images_direct(df, frame_idx, zoom):
    image_boxes = []
    used_video_indices = []

    for _, row in df.iterrows():
        img_box = load_and_box_image(row['path'], frame_idx, row['x_norm'], row['y_norm'], zoom)
        if img_box:
            image_boxes.append(img_box)
            used_video_indices.append(row['video_idx'])

    return image_boxes, used_video_indices


def load_and_box_image(img_dir, frame_idx, x, y, zoom):
    img_path = os.path.join(img_dir, f"{frame_idx}.png")
    try:
        if not os.path.exists(img_path):
            return None
        img = Image.open(img_path).convert("RGB")
        imbox = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(imbox, (x, y), frameon=False)
        return ab
    except Exception as e:
        print(f"Skipping image at {img_path} due to error: {e}")
        return None


def plot_embedding_frame_merged(
    df_frame,
    out_path,
    frame_idx,
    zoom=0.2,
    use_grid=False,
    grid_size=20
):
    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

    df_frame = normalize_tsne(df_frame)

    if use_grid:
        image_boxes, used_video_indices = place_images_on_grid(df_frame, frame_idx, zoom, grid_size)
    else:
        image_boxes, used_video_indices = place_images_direct(df_frame, frame_idx, zoom)

    for box in image_boxes:
        ax.add_artist(box)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.tight_layout()

    out_img_path = os.path.join(out_path, f"frame_{frame_idx:03d}.png")
    fig.savefig(out_img_path, dpi=300, bbox_inches='tight')
    plt.close()

    return out_img_path


def make_gif(
    output_dir,
    df,
    frame_times,
    method,
    n_videos=3000,
    n_seed=42,
    use_grid=True,
    zoom=0.2,
    grid_size=20,
    gif_duration=1000  # in milliseconds
):
    """
    Create a GIF visualizing t-SNE embedding evolution over time.

    Parameters:
        output_dir (str): Directory to save frames and GIF.
        df (DataFrame): DataFrame containing video/frame metadata and t-SNE coords.
        frame_times (int): Number of frames/timestamps to plot.
        method (str): Label for method, used in filename.
        n_videos (int): Number of videos visualized, used in filename.
        n_seed (int): Random seed used, used in filename.
        use_grid (bool): If True, use grid-based layout.
        zoom (float): Zoom level for thumbnails.
        grid_size (int): Grid resolution (if use_grid=True).
        save_selected_videos (bool): Whether to save video indices.
        gif_duration (int): Duration per frame in ms.
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_image_paths = []

    for t in range(frame_times):
        img_path = plot_embedding_frame_merged(
            df_frame=df,
            out_path=output_dir,
            frame_idx=t,
            zoom=zoom,
            use_grid=use_grid,
            grid_size=grid_size,
        )
        frame_image_paths.append(img_path)
    if use_grid:
        gif_filename = f"embedding_evolution_{method}_{n_videos}_videos_seed_{n_seed}_grid.gif"
    else:
        gif_filename = f"embedding_evolution_{method}_{n_videos}_videos_seed_{n_seed}_random.gif"
    gif_path = os.path.join(output_dir, gif_filename)

    frames = [Image.open(fname).convert("RGB") for fname in frame_image_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_duration,
        loop=0
    )

    print(f"GIF saved to {gif_path}")


# # === Main execution block ===
if __name__ == "__main__":
    # =================================
    features_dir = f"/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_features/epoch_{epoch}/"
    frame_root = "/data1/crops/random_walk_frames/IR_108_cm_years-2013/train_256"

    if aggregation:
        save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_seed_{n_seed}_{n_videos}_videos_agg_{aggregate_by}_{aggregation_mode}"
    else:
        save_filename = f"{method}_{arg_name}_{arg_value}_epoch_{epoch}_seed_{n_seed}_{n_videos}_videos"

    csv_file = os.path.join(features_dir, f"{save_filename}.csv")
    output_dir = os.path.join(features_dir, f"frames_embedding_{method}")
    os.makedirs(output_dir, exist_ok=True)

    # === LOAD EMBEDDING DATA ===
    df = pd.read_csv(csv_file, usecols=['video_idx', 'tsne_x', 'tsne_y'])

    print(f"Loaded CSV: {csv_file}, shape: {df.shape}")

    # === ATTACH FRAME IMAGE PATHS TO DATAFRAME ===
    # Step 1: Get sorted list of folder names (these are the real video names)
    video_folders = sorted(os.listdir(frame_root))
    print(f"Found {len(video_folders)} video folders.")

    # Step 2: Map video_idx to actual folder name
    df['path'] = df.apply(get_image_path, axis=1)
    print(df)

    # Get video idx from dataframe
    unique_videos = df['video_idx'].unique()
    
    #set the seed
    np.random.seed(n_seed)

    # Select n_videos unique videos
    selected_videos = sorted(np.random.choice(unique_videos, subsample_videos, replace=False))
    np.save(os.path.join(output_dir, f"selected_video_indices_{subsample_videos}_seed_{n_seed}.npy"), selected_videos)
    
    #print(len(selected_videos))
    df = df[df['video_idx'].isin(selected_videos)]
  

    make_gif(
    output_dir,
    df,
    frame_times,
    method,
    subsample_videos,
    n_seed,
    use_grid=False,
    zoom=zoom,
    grid_size=grid_size,
    gif_duration=1000  # in milliseconds
)

