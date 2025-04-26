import datetime
import os
import time
import sys
import yaml
import argparse

import numpy as np
import torch
import torchvision
from torch import nn
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torchvision.datasets.samplers.clip_sampler import RandomClipSampler, UniformClipSampler

import data
from data.kinetics import Kinetics400
from data.video import VideoList
from model import CRW
import utils


def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device, epoch, print_freq,
                    vis=None, checkpoint_fn=None):
    """
    Train the model for one full epoch.

    Args:
        model (nn.Module): The model to train.
        optimizer (Optimizer): Optimizer to update model weights.
        lr_scheduler (Scheduler): Learning rate scheduler.
        data_loader (DataLoader): DataLoader providing training batches.
        device (torch.device): Device to move inputs and model (e.g., 'cuda' or 'cpu').
        epoch (int): Current epoch number (for logging).
        print_freq (int): How often to print logs (in steps).
        vis (optional): Visualization logger (e.g., Weights & Biases). Default is None.
        checkpoint_fn (optional): Function to save model checkpoints. Default is None.

    Returns:
        None
    """
    # Set model to training mode (important for layers like BatchNorm, Dropout)
    model.train()

    # Initialize metric logger to track loss, learning rate, and clips per second
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('clips/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))

    header = f'Epoch: [{epoch}]'

    # Iterate over training data
    for step, (video, orig) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()

        # Move video batch to device (GPU/CPU)
        video = video.to(device)

        # Forward pass: get model output, loss, and diagnostics
        output, loss, diagnostics = model(video)
        loss = loss.mean()  # Average loss if needed (e.g., multi-GPU)

        # Log metrics to visualization tool (e.g., Weights & Biases) occasionally
        if vis is not None and np.random.random() < 0.01:
            vis.wandb_init(model)
            vis.log({'loss': loss.item()})
            vis.log({k: v.mean().item() for k, v in diagnostics.items()})

        # Save model checkpoint occasionally
        if checkpoint_fn is not None and np.random.random() < 0.005:
            checkpoint_fn()

        # Backpropagation: update model weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update training metrics
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['clips/s'].update(video.shape[0] / (time.time() - start_time))

        # Update learning rate
        lr_scheduler.step()

    # Save a final checkpoint at the end of the epoch
    if checkpoint_fn is not None:
        checkpoint_fn()


def _get_cache_path(filepath):
    """
    Generate a unique cache file path for a given input file.

    Args:
        filepath (str): Path of the file for which the cache path is to be created.

    Returns:
        str: Path to the cache file (under ~/.torch/vision/datasets/kinetics/).
    """
    import hashlib

    # Compute a SHA-1 hash of the file path to uniquely identify it
    h = hashlib.sha1(filepath.encode()).hexdigest()

    # Create a cache file path based on the hash (using only first 10 characters)
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "kinetics", h[:10] + ".pt")

    # Expand ~ to the user's home directory
    cache_path = os.path.expanduser(cache_path)

    return cache_path


def collate_fn(batch):
    """
    Custom collate function to process a batch of data samples by removing audio data.

    Args:
        batch (list of tuples): Each element in the batch is expected to be a tuple (video, audio).
    
    Returns:
        Tensor or a batch structure processed by default_collate, containing only the video data.
    """
    # Extract only the first element (video) from each (video, audio) tuple in the batch
    batch = [d[0] for d in batch]

    # Use PyTorch's default collate function to combine the list into a batch tensor
    return default_collate(batch)



def main(args):
    """
    Main training loop for a video self-supervised model.

    This function prepares the dataset, dataloaders, model, optimizer, and learning rate scheduler,
    and then performs training over multiple epochs, saving model checkpoints along the way.

    Args:
        args (Namespace): Parsed command line arguments containing settings for training.
    """
    
    # Print basic info about environment
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    # Set device and enable CuDNN autotuner for faster training
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # Prepare training and validation directories
    print("Preparing training dataloader")
    traindir = os.path.join(args.data_path, 'train_256' if not args.fast_test else 'val_256')
    valdir = os.path.join(args.data_path, 'val_256')

    st = time.time()  # Start time to measure dataset loading time
    cache_path = _get_cache_path(traindir)  # Path for dataset caching

    # Define data augmentations for training
    transform_train = utils.augs.get_train_transforms(args)

    def make_dataset(is_train, cached=None):
        """
        Helper to create dataset depending on the input type (video dataset, image folder, or video list).
        """
        _transform = transform_train if is_train else transform_test

        if 'kinetics' in args.data_path.lower():
            # Load Kinetics400 dataset
            return Kinetics400(
                traindir if is_train else valdir,
                frames_per_clip=args.clip_len,
                step_between_clips=1,
                transform=transform_train,
                extensions=('mp4',),
                frame_rate=args.frame_skip,
                _precomputed_metadata=cached
            )
        elif os.path.isdir(args.data_path):
            # Assume image folder structure if path is a directory
            return torchvision.datasets.ImageFolder(
                root=args.data_path,
                transform=_transform
            )
        else:
            # Load from a video list
            return VideoList(
                filelist=args.data_path,
                clip_len=args.clip_len,
                is_train=is_train,
                frame_gap=args.frame_skip,
                transform=_transform,
                random_clip=True,
            )

    # Load cached dataset if available
    if args.cache_dataset and os.path.exists(cache_path):
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
        cached = dict(
            video_paths=dataset.video_clips.video_paths,
            video_fps=dataset.video_clips.video_fps,
            video_pts=dataset.video_clips.video_pts
        )
        dataset = make_dataset(is_train=True, cached=cached)
        dataset.transform = transform_train
    else:
        dataset = make_dataset(is_train=True)
        if args.cache_dataset and 'kinetics' in args.data_path.lower():
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            dataset.transform = None  # Save dataset without transform
            torch.save((dataset, traindir), cache_path)
    
    # Recompute clips if necessary
    if hasattr(dataset, 'video_clips'):
        dataset.video_clips.compute_clips(args.clip_len, 1, frame_rate=args.frame_skip)

    print("Dataset preparation took", time.time() - st, "seconds")

    def make_data_sampler(is_train, dataset):
        """
        Helper to create a sampler for dataloading.
        """
        torch.manual_seed(0)
        if hasattr(dataset, 'video_clips'):
            _sampler = RandomClipSampler
            return _sampler(dataset.video_clips, args.clips_per_video)
        else:
            return torch.utils.data.sampler.RandomSampler(dataset) if is_train else None

    # Create DataLoader
    print("Creating data loaders")
    train_sampler = make_data_sampler(True, dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers // 2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Initialize visualization tool (wandb/visdom)
    vis = utils.visualize.Visualize(args) if args.visualize else None

    # Create the model
    print("Creating model")
    model = CRW(args, vis=vis).to(device)
    print(model)

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_milestones = [len(data_loader) * m for m in args.lr_milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.lr_gamma
    )

    model_without_ddp = model  # Handle model in case of DataParallel
    if args.data_parallel:
        model = torch.nn.parallel.DataParallel(model)
        model_without_ddp = model.module

    # Optionally load a partial checkpoint
    if args.partial_reload:
        checkpoint = torch.load(args.partial_reload, map_location='cpu')
        utils.partial_load(checkpoint['model'], model_without_ddp)
        optimizer.param_groups[0]["lr"] = args.lr
        args.start_epoch = checkpoint['epoch'] + 1

    # Optionally resume training from full checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    # Define function to save model checkpoints
    def save_model_checkpoint():
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # Start training loop
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(
            model, optimizer, lr_scheduler, data_loader,
            device, epoch, args.print_freq,
            vis=vis, checkpoint_fn=save_model_checkpoint
        )

    # Print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="Path to the YAML config file")
    # Allow override from CLI
    parser.add_argument('--batch-size', type=int, help="Batch size")
    parser.add_argument('--workers', type=int, help="Number of workers")
    parser.add_argument('--data-path', type=str, help="Path to dataset")
    parser.add_argument('--epochs', type=int, help="Number of epochs")
    parser.add_argument('--lr', type=float, help="Learning rate")
    # Add more if you want overrideable params
    return parser.parse_args()

def load_config(args):
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Override YAML values with CLI args if provided
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.workers is not None:
        config['workers'] = args.workers
    if args.data_path is not None:
        config['data_path'] = args.data_path
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['lr'] = args.lr

    return config

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)

    # Simulate the original `args` object
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    final_args = Args(**config)

    # Save the config before training
    if hasattr(final_args, 'output_dir') and final_args.output_dir:
        os.makedirs(final_args.output_dir, exist_ok=True)
        config_save_path = os.path.join(final_args.output_dir, 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.safe_dump(vars(final_args), f)
        print(f"Saved config to {config_save_path}")

    main(final_args)


# if __name__ == "__main__":
#     args = utils.arguments.train_args()
#     main(args)
