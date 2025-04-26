import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data
import random
import cv2
from PIL import Image

import torchvision.transforms as transforms


class VideoList(data.Dataset):
    """
    Dataset for loading video clips from folders of frames.
    
    Each sample is a clip consisting of `clip_len` frames, optionally sampled with a frame gap.
    
    Args:
        filelist (str): Path to the file containing folder paths and their frame counts.
        clip_len (int): Number of frames in each sampled clip.
        is_train (bool): Whether the dataset is used for training (currently unused).
        frame_gap (int): Number of frames to skip between each frame.
        transform (callable, optional): A function/transform to apply to the sampled frames.
        random_clip (bool): Whether to randomly select the start frame for clips.
    """
    def __init__(self, filelist, clip_len, is_train=True, frame_gap=1, transform=None, random_clip=True):
        self.filelist = filelist
        self.clip_len = clip_len
        self.is_train = is_train
        self.frame_gap = frame_gap
        self.transform = transform
        self.random_clip = random_clip

        self.jpgfiles = []  # List of folder paths
        self.fnums = []     # List of corresponding number of frames

        # Read filelist.txt and parse folder paths and frame counts
        with open(self.filelist, 'r') as f:
            for line in f:
                rows = line.strip().split()
                folder_path = rows[0]
                num_frames = int(rows[1])

                self.jpgfiles.append(folder_path)
                self.fnums.append(num_frames)

    def __getitem__(self, index):
        """
        Sample a clip from the video at the given index.
        """
        index = index % len(self.jpgfiles)  # Make dataset "circular"
        folder_path = self.jpgfiles[index]
        total_frames = self.fnums[index]

        frame_gap = self.frame_gap
        start_frame = 0
        readjust = False

        # Adjust frame gap if clip cannot fit into total frames
        while total_frames - self.clip_len * frame_gap < 0 and frame_gap > 1:
            frame_gap -= 1
            readjust = True

        if readjust:
            print(f"⚠️ Frame gap adjusted to {frame_gap} for {folder_path}")

        # Determine the start frame
        max_start = total_frames - self.clip_len * frame_gap
        if self.random_clip:
            start_frame = random.randint(0, max(0, max_start))
        else:
            start_frame = 0

        # Read frame filenames and sort numerically
        frame_files = os.listdir(folder_path)
        frame_files.sort(key=lambda x: int(x.split('.')[0]))

        imgs = []

        # Load and stack the selected frames
        for i in range(self.clip_len):
            idx = start_frame + i * frame_gap
            img_path = os.path.join(folder_path, frame_files[idx])

            # Read image and convert from BGR to RGB
            img = cv2.imread(img_path)[:, :, ::-1]
            imgs.append(img)

        imgs = np.stack(imgs)

        # Apply optional transformations
        if self.transform is not None:
            imgs = self.transform(imgs)

        # Return the clip along with dummy labels (placeholders)
        return imgs, torch.tensor(0), torch.tensor(0)

    def __len__(self):
        """
        Total number of video clips in the dataset.
        """
        return len(self.jpgfiles)


class SingleVideoDataset(data.Dataset):
    """
    Dataset for sampling multiple clips from a single in-memory video array.
    
    Args:
        video (np.ndarray): Video as a numpy array (frames x H x W x C).
        clip_len (int): Number of frames in each clip.
        fps_range (list of int): Range of FPS to randomly sample between.
        n_clips (int): Total number of clips to sample.
    """
    def __init__(self, video, clip_len, fps_range=[1, 1], n_clips=100000):
        self.video = video
        self.clip_len = clip_len
        self.fps_range = fps_range
        self.n_clips = n_clips

    def __getitem__(self, index):
        """
        Randomly sample a clip from the video at a random FPS within the specified range.
        """
        fps = np.random.randint(self.fps_range[0], self.fps_range[1] + 1)
        max_idx = self.video.shape[0] // fps - self.clip_len

        idx = np.random.randint(0, max_idx)
        clip = self.video[::fps][idx:idx + self.clip_len]

        return clip

    def __len__(self):
        """
        Total number of clips available in this dataset (virtual).
        """
        return self.n_clips