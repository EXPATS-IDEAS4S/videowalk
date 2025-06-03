import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import sys

sys.path.append('/home/Daniele/codes/videowalk/code/check_training')
from plot_video_embeddings import make_gif


def plot_embedding_frame_old(df_frame, out_path, grid_size=10, zoom=0.1):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    x = df_frame['component_1'].values
    y = df_frame['component_2'].values
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
            img = Image.open(row['video_path']).convert("RGB")
            #img = ImageOps.invert(img)
            imbox = OffsetImage(img, zoom=zoom)
            ab = AnnotationBbox(imbox, (centers[i], centers[j]), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"Warning: Skipping image at {row['path']} due to error: {e}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    #ax.invert_yaxis()
    ax.axis('off')
    plt.tight_layout()
    
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return frame_filename

def make_gif_old(frame_image_paths, gif_path, duration=1000):
    """
    Create a GIF from a list of image paths.
    """
    
    # Load all frames
    frames = [Image.open(fname).convert("RGB") for fname in frame_image_paths]

    # Save as GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,  # 2000 ms = 2 seconds per frame
        loop=0  # loop forever
    )

    print(f"GIF saved to {gif_path}")



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

    
    df = pd.read_csv(output_csv_path)#, usecols=['video_idx', 'tsne_x', 'tsne_y'])
    #print(df)
    # change component_1 and component_2 to tsne_x and tsne_y
    df.rename(columns={'component_1': 'tsne_x', 'component_2': 'tsne_y'}, inplace=True)
    
    #add one column with video idx
    df['video_idx'] = df.index
    #print(df)

    # Get sorted list of folder names (these are the real video names)
    video_folders = sorted(os.listdir(dir_vid))
    print(f"Found {len(video_folders)} video folders.")

    # Add paths to dataframe
    df['path'] = df['video_idx'].apply(lambda idx: os.path.join(dir_vid, video_folders[idx]))
    
    # Run steps
    if from_selected_indices:
        #Open selected index from npy file
        selected_vid_dir = f'/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_features/epoch_29/frames_embedding_tSNE/'
        selected_indices = np.load(os.path.join(selected_vid_dir, f"selected_video_indices_{subsample_videos}_seed_{n_seed}.npy"))
        #print(f"Using selected indices: {selected_indices}")
        
        #filter datsframe for selected indices
        df = df[df['video_idx'].isin(selected_indices)]
        #print(df)
    else:
        # Get video idx from dataframe
        unique_videos = df['video_idx'].unique()
        
        #set the seed
        np.random.seed(n_seed)

        # Select n_videos unique videos
        selected_videos = sorted(np.random.choice(unique_videos, subsample_videos, replace=False))
        np.save(os.path.join(main_dir, f"selected_video_indices_{subsample_videos}_seed_{n_seed}.npy"), selected_videos)
    
        #print(len(selected_videos))
        df = df[df['video_idx'].isin(selected_videos)]
  

    make_gif(
    main_dir,
    df,
    frame_times=n_frames,
    method='tSNE-affinity',
    n_videos=subsample_videos,
    n_seed=n_seed,
    use_grid=False,
    zoom=zoom,
    grid_size=grid_size,
    gif_duration=1000  # in milliseconds
)

    
    
"""    
    frames_path_list = []
    for frame_idx in range(n_frames):
        # Construct frame-specific filename
        frame_filename = f"{frame_idx:01d}.png"
        
        #open frame with video paths
        df_frame = pd.read_csv(output_csv_path)
        print(df_frame)

        if from_selected_indices:
            #add a column for video idx 
            df_frame["video_idx"] = df_frame.index
            
            #select the video_idx corresponding to the selected indices
            df_frame = df_frame[df_frame["video_idx"].isin(selected_indices)]
            print(df_frame)

        # Add the frame filename to the path for matching
        df_frame["video_path"] = df_frame["video_path"].apply(lambda p: os.path.join(p, frame_filename))
        #print(df_frame["video_path"].tolist())

        out_frame_dir = f"{main_dir}frames_embedding/"
        os.makedirs(out_frame_dir, exist_ok=True)
        out_path = f"{out_frame_dir}tsne_perp-{perplexity}_frame-{frame_idx}.png"
        frames_path_list.append(out_path)
        print(f"Plotting frame {frame_idx} with {len(df_frame)} entries.")

        plot_embedding_frame(df_frame, out_path, grid_size, zoom)
    
    #make gif
    if from_selected_indices:
        filename_gif = main_dir+'embedding_video_from_selected_indeces.gif'
    else:
        filename_gif = main_dir+'embedding_video.gif'
    
    make_gif( frames_path_list, filename_gif)
"""