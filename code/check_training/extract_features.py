import torch
import os
import json
import numpy as np
from tqdm import tqdm
import sys

#APPEND PATH
sys.path.append('/home/Daniele/codes/videowalk/code/')

from model import CRW
from data.video import VideoList
import utils

# ------------------ Configuration ------------------
epoch = 29
class Args:
    resume = f'/home/Daniele/codes/videowalk/code/checkpoints/first_run/model_{epoch}.pth'  # Your trained checkpoint
    data_path = '/home/Daniele/codes/videowalk/code/checkpoints/first_run/filelist.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type = 'scratch'
    img_size = 100
    clip_len = 8
    frame_skip = 1
    batch_size = 1  # Keep as 1 for metadata clarity
    workers = 4
    output_dir = f'/data1/fig/videowalk/run-1_30-epochs_8k-clips/extracted_features/epoch_{epoch}'
    no_l2 = False
    remove_layers =  ['layer4']   

args = Args()
os.makedirs(args.output_dir, exist_ok=True)

# ------------------ Load Model ------------------
model = CRW(args).to(args.device)
checkpoint = torch.load(args.resume, map_location=args.device)
utils.partial_load(checkpoint['model'], model, skip_keys=['head'])
model.eval()

# ------------------ Load Dataset ------------------
_transform = lambda x: x  # No transform for raw features
video_dataset = VideoList(
    filelist=args.data_path,
    clip_len=args.clip_len,
    is_train=False,
    frame_gap=args.frame_skip,
    transform=_transform,
    random_clip=False
)
dataloader = torch.utils.data.DataLoader(
    video_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True
)

# ------------------ Feature Extraction ------------------
all_features = []
metadata = []

with torch.no_grad():
    for video_idx, (imgs, imgs_orig, meta) in enumerate(tqdm(dataloader)):
        # Normalize and permute images
        imgs = imgs.float() / 255.0  # now in float32 and scaled to [0, 1]
        imgs = imgs.permute(0, 1, 4, 2, 3).permute(0, 2, 1, 3, 4)

        # Now pass to encoder
        feats = model.encoder(imgs.to(args.device))
        #feats = model.encoder(imgs.transpose(1, 2))  # [1, D, T, H', W']
        if not args.no_l2:
            feats = torch.nn.functional.normalize(feats, dim=1)

        D, T, H, W = feats.shape[1:]
        feats = feats.squeeze(0)  # Remove batch

        for t in range(T):
            frame_feat = feats[:, t]  # [D, H, W]
            all_features.append(frame_feat.view(D, -1).T.cpu().numpy())  # [H*W, D]

            for i in range(H):
                for j in range(W):
                    metadata.append({
                        'video_idx': video_idx,
                        'frame_idx': t,
                        'patch_row': i,
                        'patch_col': j
                    })

# ------------------ Save Features and Metadata ------------------
features_np = np.concatenate(all_features, axis=0)
np.save(os.path.join(args.output_dir, 'features.npy'), features_np)

with open(os.path.join(args.output_dir, 'features_metadata.json'), 'w') as f:
    json.dump(metadata, f)

print(f"Saved {features_np.shape[0]} features of dimension {features_np.shape[1]}")