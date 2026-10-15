import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def compute_mean_std(root_dir, output_dir, valid_exts={'.png'}):
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    for subdir, _, files in os.walk(root_dir):
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        for fname in tqdm(image_files, desc=f"Processing {subdir}"):
            path = os.path.join(subdir, fname)
            with Image.open(path) as img:
                img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
                pixel_sum += img.sum()
                pixel_squared_sum += (img ** 2).sum()
                pixel_count += img.size

    if pixel_count == 0:
        raise ValueError("No images found.")

    mean = pixel_sum / pixel_count
    std = np.sqrt((pixel_squared_sum / pixel_count) - (mean ** 2))

    print(f"\nMean: {mean:.6f}")
    print(f"Std: {std:.6f}")

    #save mean and std to a file in the current directory
    output_path = os.path.join(output_dir, 'mean_std.txt')
    with open(output_path, 'w') as f:
        f.write(f"Mean: {mean:.6f}\n")
        f.write(f"Std: {std:.6f}\n")
    print(f"Mean and Std saved to {output_path}")
    

# Example usage:
out_dir = '/data1/runs/videowalk/4_3x3_8_100x100_108-CMA_70k_500/'
vid_dir = '/data1/crops/random_walk_frames/IR_108_cm_years-2013-2014-2015-2016-2017-2018-2019-2020/train_256'
compute_mean_std(vid_dir, out_dir)


# nohup 2845450