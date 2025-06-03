import glob
from PIL import Image
import imageio
import os
import matplotlib.pyplot as plt


def create_plots_and_gif(video_paths_list, index, output_dir='output'):
    if index >= len(video_paths_list):
        print(f"Index {index} out of bounds for video_paths_list.")
        return

    video_dir = video_paths_list[index]
    frame_paths = sorted(
        glob.glob(os.path.join(video_dir, "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    if not frame_paths:
        print(f"No frames found in {video_dir}")
        return

    # Create frame output directory
    frame_output_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frame_output_dir, exist_ok=True)

    plotted_frames = []

    for i, frame_path in enumerate(frame_paths):
        img = Image.open(frame_path).convert("RGB")
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()

        plot_path = os.path.join(frame_output_dir, f"video_{index}_frame_{i:03d}.png")
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close()
        plotted_frames.append(plot_path)

    # Create GIF from plotted frame images
    images = [Image.open(p) for p in plotted_frames]
    gif_path = os.path.join(output_dir, f"video_{index}.gif")
    imageio.mimsave(gif_path, images, duration=1000)
    print(f"Saved GIF to {gif_path}")


epoch = 29
video_dir = '/data1/crops/random_walk_frames/IR_108_cm_years-2013/train_256/'
output_dir = f'/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/extracted_affinities/epoch_{epoch}'
video_list = [2286,1824,1679,1424,520,409,8235,5875,5274,4596,4581,3626,2745,2196,883,333]


list_videos = sorted(glob.glob(os.path.join(video_dir, '*')))
print(len(list_videos))

for video_idx in video_list:
    create_plots_and_gif(list_videos, video_idx, output_dir)