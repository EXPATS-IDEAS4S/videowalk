import torch

# Define dummy class to handle loading of custom object 'Args'
class Args:
    pass

dir_path = '/home/Daniele/codes/videowalk/code/checkpoints/first_run/'
checkpoint = torch.load(f"{dir_path}model_1.pth", map_location='cpu')

# Print keys
print("Checkpoint keys:", checkpoint.keys())

# Print content summaries
print("\nEpoch:", checkpoint['epoch'])

print("\nModel state_dict (first 5 keys):")
for k in list(checkpoint['model'].keys())[:5]:
    print(f"  {k}: {checkpoint['model'][k].shape}")

print("\nOptimizer state keys:", checkpoint['optimizer'].keys())

print("\nLR Scheduler state:", checkpoint['lr_scheduler'])

print("\nArgs (type and content):")
print(type(checkpoint['args']))
print(vars(checkpoint['args']) if hasattr(checkpoint['args'], '__dict__') else checkpoint['args'])

