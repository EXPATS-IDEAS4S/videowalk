import os

def generate_filelist(root_dir,
                      output_file='filelist.txt',
                      valid_exts={'.jpg', '.png'}):
    """
    Generate a filelist.txt where each line is:
    <folder_path> <number_of_frames>
    
    Args:
        root_dir (str): Path containing subfolders with extracted frames.
        output_file (str): Where to save the filelist.
        valid_exts (set): Set of allowed frame file extensions.
    """
    folder_list = []

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path):
            # Count valid image frames in the folder
            frames = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[-1].lower() in valid_exts
            ]
            num_frames = len(frames)

            if num_frames > 0:
                folder_list.append((folder_path, num_frames))
            else:
                print(f"Warning: No valid frames found in {folder_path}")

    # Write filelist
    with open(output_file, 'w') as f:
        for folder_path, num_frames in folder_list:
            f.write(f"{folder_path} {num_frames}\n")

    print(f"Filelist saved to: {output_file}")

# Run the function
root_dir='/data1/crops/random_walk_frames/IR_108/train_256'
output_file='filelist.txt'
valid_exts={'.jpg', '.png'}
generate_filelist(root_dir, output_file,valid_exts)


