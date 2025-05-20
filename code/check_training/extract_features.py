import torch
import os
import json
import numpy as np
from tqdm import tqdm
import sys
from types import SimpleNamespace
import yaml

#APPEND PATH
sys.path.append('/home/Daniele/codes/videowalk/code/')

from model import CRW
from data.video import VideoList
import utils

# ------------------ Load Configuration from YAML ------------------
config_path = '/home/Daniele/codes/videowalk/code/configs/config_test.yaml'  # <<< CHANGE THIS TO YOUR CONFIG FILE
with open(config_path, 'r') as f:
    config_dict = yaml.safe_load(f)

# Convert dict to object-like access
args = SimpleNamespace(**config_dict)

# If resume or output_dir depends on epoch, adjust dynamically
if 'epoch' in config_dict:
    args.resume = f'/home/Daniele/codes/videowalk/code/checkpoints/first_run/model_{args.epoch}.pth'
    args.output_dir = f'/data1/fig/videowalk/run-1_30-epochs_8k-clips/extracted_features/epoch_{args.epoch}'

os.makedirs(args.output_dir, exist_ok=True)

# ------------------ Load Model ------------------

class Args:
    pass

model = CRW(args).to(args.device)
checkpoint = torch.load(args.resume, map_location=args.device)
utils.partial_load(checkpoint['model'], model, skip_keys=['head'])
#model.load_state_dict(checkpoint['model']) # Load the model state dict if not using partial load
model.eval()

# ------------------ Load Dataset ------------------
#_transform = lambda x: x  
transform_train = utils.augs.get_train_transforms(args)
video_dataset = VideoList(
    filelist=args.data_path,
    clip_len=args.clip_len,
    is_train=False,
    frame_gap=args.frame_skip,
    transform=transform_train, 
    random_clip=False
)

#TODO: create patching function

dataloader = torch.utils.data.DataLoader(
    video_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers // 2,
    pin_memory=True
)


# ------------------ Feature Extraction ------------------
all_features = []
metadata = []

with torch.no_grad():
    for video_idx, (imgs, imgs_orig, meta) in enumerate(tqdm(dataloader)):
        imgs = imgs[0]  # Since it's a list of 1 tensor: (T, C*N, H, W)
        #print(imgs.shape)    

        B, T, cn, H, W = imgs.shape
        C = 3
        N = cn // C
        assert cn % C == 0, "Channel dimension not divisible by C=3"

        # Reshape to (B, T, N, C, H, W), then permute to (B, N, C, T, H, W)
        imgs = imgs.view(B, T, N, C, H, W)       # (B, T, N, C, H, W)
        imgs = imgs.permute(0, 2, 3, 1, 4, 5)    # (B, N, C, T, H, W)
        #print(imgs.shape)

        # Now imgs is (1, N, C, T, H, W)
        feats, maps = model.pixels_to_nodes(imgs.to(args.device))
        #print(feats.shape)
        #print(maps.shape)

        # Remove batch dimension
        feats = feats.squeeze(0)  # shape: (C, T, N)
        #print(feats.shape)

        C, T, N = feats.shape

        for t in range(T):  # for each time step
            for n in range(N):  # for each node (patch/image)
                node_feat = feats[:, t, n]  # shape: (C,)\
                node_feat = node_feat.unsqueeze(0)
                #print(node_feat.shape)
                all_features.append(node_feat.cpu().numpy())
                #print(all_features)

                metadata.append({
                    'video_idx': video_idx,
                    'frame_idx': t,
                    'node_idx': n
                })


# ------------------ Save Features and Metadata ------------------
features_np = np.concatenate(all_features, axis=0)
print(f"Extracted features shape: {features_np.shape}")
np.save(os.path.join(args.output_dir, 'features.npy'), features_np)

with open(os.path.join(args.output_dir, 'features_metadata.json'), 'w') as f:
    json.dump(metadata, f)

#print(f"Saved features")