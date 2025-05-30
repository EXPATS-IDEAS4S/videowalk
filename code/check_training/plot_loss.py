import re
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths
log_file = '/home/Daniele/codes/videowalk/code/nohup.out'
out_dir = '/data1/runs/videowalk/3_3x3_patches_8_frames_100x100_pixels_108-CMA_channel/'

# Regex to extract epoch, loss, and lr
pattern = re.compile(r"Epoch: \[(\d+)]\s+\[\d+/\d+].*?lr: ([\deE\.-]+).*?loss: ([\d\.]+)")

# Storage
epoch_losses = defaultdict(list)
epoch_lrs = defaultdict(list)

# Parse the file
with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            lr = float(match.group(2))
            loss = float(match.group(3))
            epoch_losses[epoch].append(loss)
            epoch_lrs[epoch].append(lr)

# Averages
epochs = sorted(epoch_losses.keys())
avg_losses = [sum(epoch_losses[e]) / len(epoch_losses[e]) for e in epochs]
avg_lrs = [sum(epoch_lrs[e]) / len(epoch_lrs[e]) for e in epochs]

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Loss curve
ax1.plot(epochs, avg_losses, 'b-o', label='Loss')
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', color='blue', fontsize=14)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# LR curve
ax2 = ax1.twinx()
ax2.plot(epochs, avg_lrs, 'r--o', label='Learning Rate')
ax2.set_ylabel('Learning Rate', color='red', fontsize=14)
ax2.tick_params(axis='y', labelcolor='red')

# Title and save
plt.title('Average Loss and Learning Rate per Epoch', fontsize=15, fontweight='bold')
fig.tight_layout()
plt.savefig(f'{out_dir}loss_and_lr_per_epoch.png', bbox_inches='tight')
