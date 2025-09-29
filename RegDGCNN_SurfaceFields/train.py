# train.py
"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu

Training script for RegDGCNN pressure field prediction model on the DrivAerNet++ dataset.
This version includes distributed training support for multi-GPU acceleration.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import heapq
import re

# Import modules
from data_loader import get_dataloaders, PRESSURE_MEAN, PRESSURE_STD
from model_pressure import RegDGCNN_pressure
from utils import setup_logger, setup_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train pressure prediction models on DrivAerNet++')

    # Basic settings
    parser.add_argument('--exp_name', type=str, default='PressurePrediction', help='Experiment name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Data settings
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--subset_dir', type=str, required=True, help='Path to train/val/test splits')
    parser.add_argument('--cache_dir', type=str, help='Path to cache directory')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of points to sample')

    # Training settings
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--test_only', action='store_true', help='Only test the model, no training')
    parser.add_argument('--gpus', type=str, default='0', help='GPUs to use (comma-separated)')
    parser.add_argument('--num_best_models', type=int, default=10, help='Number of best models to keep')

    # Model settings
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=40, help='Number of nearest neighbors')
    parser.add_argument('--output_channels', type=int, default=1, help='Number of output channels')

    return parser.parse_args()


def initialize_model(args, local_rank):
    """Initialize and return the RegDGCNN model."""
    args = vars(args)

    model = RegDGCNN_pressure(args).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True, output_device=local_rank
    )
    return model


def train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for data, targets in tqdm(train_dataloader, desc="[Training]"):
        data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
        targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(1), targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def validate(model, val_dataloader, criterion, local_rank):
    """Validate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, targets in tqdm(val_dataloader, desc="[Validation]"):
            data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
            targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs = model(data)
            loss = criterion(outputs.squeeze(1), targets)
            total_loss += loss.item()

    return total_loss / len(val_dataloader)


def test_model(model, test_dataloader, criterion, local_rank, exp_dir):
    """Test the model and calculate metrics."""
    model.eval()
    total_mse, total_mae = 0, 0
    total_rel_l2, total_rel_l1 = 0, 0
    total_inference_time = 0
    total_samples = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for data, targets in tqdm(test_dataloader, desc="[Testing]"):
            start_time = time.time()

            data, targets = data.squeeze(1).to(local_rank), targets.squeeze(1).to(local_rank)
            normalized_targets = (targets - PRESSURE_MEAN) / PRESSURE_STD

            outputs = model(data)
            normalized_outputs = outputs.squeeze(1)

            inference_time = time.time() - start_time
            total_inference_time += inference_time

            # Calculate metrics
            mse = criterion(normalized_outputs, normalized_targets)
            mae = F.l1_loss(normalized_outputs, normalized_targets)

            # Calculate relative errors
            rel_l2 = torch.mean(torch.norm(normalized_outputs - normalized_targets, p=2, dim=-1) /
                                torch.norm(normalized_targets, p=2, dim=-1))
            rel_l1 = torch.mean(torch.norm(normalized_outputs - normalized_targets, p=1, dim=-1) /
                                torch.norm(normalized_targets, p=1, dim=-1))

            batch_size = targets.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_rel_l2 += rel_l2.item() * batch_size
            total_rel_l1 += rel_l1.item() * batch_size
            total_samples += batch_size

            # Store normalized predictions and targets for R² calculation
            all_outputs.append(normalized_outputs.cpu())
            all_targets.append(normalized_targets.cpu())

    # Aggregate results across all processes
    total_mse_tensor = torch.tensor(total_mse).to(local_rank)
    total_mae_tensor = torch.tensor(total_mae).to(local_rank)
    total_rel_l2_tensor = torch.tensor(total_rel_l2).to(local_rank)
    total_rel_l1_tensor = torch.tensor(total_rel_l1).to(local_rank)
    total_samples_tensor = torch.tensor(total_samples).to(local_rank)

    dist.reduce(total_mse_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_mae_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_rel_l2_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_rel_l1_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(total_samples_tensor, dst=0, op=dist.ReduceOp.SUM)

    if local_rank == 0:
        # Calculate aggregated metrics
        avg_mse = total_mse_tensor.item() / total_samples_tensor.item()
        avg_mae = total_mae_tensor.item() / total_samples_tensor.item()
        avg_rel_l2 = total_rel_l2_tensor.item() / total_samples_tensor.item()
        avg_rel_l1 = total_rel_l1_tensor.item() / total_samples_tensor.item()

        # Calculate R² score - only on rank 0 with locally collected data
        all_outputs = torch.cat(all_outputs, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        ss_res = np.sum((all_targets - all_outputs) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate max MAE
        max_mae = np.max(np.abs(all_targets - all_outputs))

        print(f"Test MSE: {avg_mse:.6f}, Test MAE: {avg_mae:.6f}, Max MAE: {max_mae:.6f}, Test R²: {r_squared:.4f}")
        print(f"Relative L2 Error: {avg_rel_l2:.6f}, Relative L1 Error: {avg_rel_l1:.6f}")
        print(f"Total inference time: {total_inference_time:.2f}s for {total_samples_tensor.item()} samples")

        # Save metrics to a text file
        metrics_file = os.path.join(exp_dir, 'test_metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Test MSE: {avg_mse:.6f}\n")
            f.write(f"Test MAE: {avg_mae:.6f}\n")
            f.write(f"Max MAE: {max_mae:.6f}\n")
            f.write(f"R² Score: {r_squared:.6f}\n")
            f.write(f"Relative L2 Error: {avg_rel_l2:.6f}\n")
            f.write(f"Relative L1 Error: {avg_rel_l1:.6f}\n")
            f.write(f"Total inference time: {total_inference_time:.2f}s for {total_samples_tensor.item()} samples\n")


def train_and_evaluate(rank, world_size, args):
    """Main function for distributed training and evaluation."""
    setup_seed(args.seed)

    # Initialize process group for DDP
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    local_rank = rank
    torch.cuda.set_device(local_rank)

    # Set up logging (only on rank 0)
    if local_rank == 0:
        exp_dir = os.path.join('experiments', args.exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        log_file = os.path.join(exp_dir, 'training.log')
        setup_logger(log_file)
        logging.info(f"Arguments: {args}")
        logging.info(f"Starting training with {world_size} GPUs")
        print(f"Starting training with {world_size} GPUs")

    # Helper to restore training state from log
    def parse_log_losses(log_file, exp_dir):
        train_losses, val_losses = [], []
        best_models = []
        if not os.path.exists(log_file):
            return train_losses, val_losses, best_models

        with open(log_file, 'r') as f:
            for line in f:
                if "Epoch" in line and "Train Loss" in line and "Val Loss" in line:
                    # Example line:
                    # 2025-09-29 12:16:12,921 - INFO - Epoch 1/60 - Train Loss: 0.293800, Val Loss: 0.700069
                    try:
                        epoch_part = line.split("Epoch")[1].split("-")[0].strip()
                        # "1/60" → take first number
                        epoch = int(epoch_part.split("/")[0])
                        train_loss = float(line.split("Train Loss:")[1].split(",")[0].strip())
                        val_loss = float(line.split("Val Loss:")[1].strip())
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                    except Exception as e:
                        print(f"Warning: failed to parse line: {line.strip()} ({e})")

                if "Saved model from epoch" in line or "Replaced" in line:
                    # Example line:
                    # 2025-09-29 12:16:12,939 - INFO - Saved model from epoch 1 with Val Loss: 0.700069
                    tokens = line.strip().split()
                    try:
                        epoch = int(tokens[6])  # after "epoch"
                        val_loss = float(tokens[-1])
                        model_path = os.path.join(exp_dir, f"model_epoch{epoch}.pth")
                        heapq.heappush(best_models, (-val_loss, model_path))
                    except Exception as e:
                        print(f"Warning: failed to parse model line: {line.strip()} ({e})")

        return train_losses, val_losses, best_models

    # Initialize model
    model = initialize_model(args, local_rank)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params}")
        print(f"Total trainable parameters: {total_params}")

    # Prepare DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        args.dataset_path, args.subset_dir, args.num_points,
        args.batch_size, world_size, rank, args.cache_dir,
        args.num_workers
    )

    # Log dataset info
    if local_rank == 0:
        logging.info(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")
        print(
            f"Data loaded: {len(train_dataloader)} training batches, {len(val_dataloader)} validation batches, {len(test_dataloader)} test batches")

    # Set up criterion, optimizer, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

    best_model_path = os.path.join('experiments', args.exp_name, 'best_model.pth')
    final_model_path = os.path.join('experiments', args.exp_name, 'final_model.pth')

    # Check if test_only and model exists
    if args.test_only and os.path.exists(best_model_path):
        if local_rank == 0:
            logging.info("Loading best model for testing only")
            print("Testing the best model:")
        model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
        test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))
        dist.destroy_process_group()
        return

    # Training tracking
    # best_val_loss = float('inf')
    best_models = []
    train_losses = []
    val_losses = []
    num_best = args.num_best_models
    start_epoch = 0

    train_losses, val_losses, best_models = parse_log_losses(log_file, exp_dir)

    ckpts = [f for f in os.listdir(exp_dir) if f.endswith('.pth') and f.startswith('model_epoch')]

    if ckpts:
        latest_ckpt = max(ckpts, key=lambda f: int(f.split('epoch')[1].split('.pth')[0]))
        latest_epoch = int(latest_ckpt.split('epoch')[1].split('.pth')[0])

        model.load_state_dict(torch.load(os.path.join(exp_dir, latest_ckpt),
                                         map_location=f'cuda:{local_rank}'))
        start_epoch = latest_epoch

        # checkpoint = torch.load(os.path.join(exp_dir, latest_ckpt), map_location=f'cuda:{local_rank}')
        # model.load_state_dict(checkpoint['model_state'])
        # optimizer.load_state_dict(checkpoint['optimizer_state'])
        # scheduler.load_state_dict(checkpoint['scheduler_state'])
        # start_epoch = checkpoint['epoch']

        if local_rank == 0:
            logging.info(f"Resumed from {latest_ckpt} at epoch {start_epoch}")
            print(f"Resumed from {latest_ckpt} at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Set epoch for the DistributedSampler
        train_dataloader.sampler.set_epoch(epoch)

        # Training
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, local_rank)

        # Validation
        val_loss = validate(model, val_dataloader, criterion, local_rank)

        # Record losses
        if local_rank == 0:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            logging.info(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # # Save the best model
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), best_model_path)
            #     logging.info(f"New best model saved with Val Loss: {best_val_loss:.6f}")

            # Save top $num_best models
            model_path = os.path.join('experiments', args.exp_name, f'model_epoch{epoch+1}.pth')
            if len(best_models) < num_best:
                torch.save(model.state_dict(), model_path)
                # torch.save({'epoch': epoch + 1,  # save next epoch index
                #             'model_state': model.state_dict(),
                #             'optimizer_state': optimizer.state_dict(),
                #             'scheduler_state': scheduler.state_dict(),
                #             'val_loss': val_loss,
                #             }, model_path)
                heapq.heappush(best_models, (-val_loss, model_path))  # use -val_loss for max-heap
                logging.info(f"Saved model from epoch {epoch+1} with Val Loss: {val_loss:.6f}")
            else:
                worst_val_loss, worst_path = best_models[0]  # worst in heap
                if val_loss < -worst_val_loss:
                    torch.save(model.state_dict(), model_path)
                    # torch.save({'epoch': epoch + 1,  # save next epoch index
                    #         'model_state': model.state_dict(),
                    #         'optimizer_state': optimizer.state_dict(),
                    #         'scheduler_state': scheduler.state_dict(),
                    #         'val_loss': val_loss,
                    #         }, model_path)
                    heapq.heapreplace(best_models, (-val_loss, model_path))
                    logging.info(f"Replaced {worst_path} with epoch {epoch+1} model (Val Loss: {val_loss:.6f})")
                    if os.path.exists(worst_path):
                        os.remove(worst_path)

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Save progress plot
            if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                plt.figure(figsize=(10, 5))
                epochs_range = range(1, len(train_losses) + 1)
                plt.plot(epochs_range, train_losses, label='Training Loss')
                plt.plot(epochs_range, val_losses, label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title(f'Training Progress - RegDGCNN')
                plt.savefig(os.path.join('experiments', args.exp_name, 'training_progress.png'))
                plt.close()

    # Save final model
    if local_rank == 0:
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        print(f"Final model saved to {final_model_path}")

    # Make sure all processes sync up before testing
    dist.barrier()

    # === Test all saved models ===
    if local_rank == 0:
        logging.info("Testing all saved models")
        print("Testing all saved models:")

        ckpt_files = sorted(
            [f for f in os.listdir(exp_dir) if f.endswith(".pth")],
            key=lambda x: os.path.getmtime(os.path.join(exp_dir, x))  # sort by save time
        )

        for ckpt in ckpt_files:
            ckpt_path = os.path.join(exp_dir, ckpt)
            # checkpoint = torch.load(ckpt_path, map_location=f'cuda:{local_rank}')
            # model.load_state_dict(checkpoint['model_state'])
            model.load_state_dict(torch.load(ckpt_path, map_location=f'cuda:{local_rank}'))


            logging.info(f"Testing {ckpt}")
            print(f"Testing {ckpt}:")
            test_model(model, test_dataloader, criterion, local_rank, exp_dir)


    # Test the final model
    if local_rank == 0:
        logging.info("Testing the final model")
        print("Testing the final model:")
    test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))

    # # Test the best model
    # if local_rank == 0:
    #     logging.info("Testing the best model")
    #     print("Testing the best model:")
    # model.load_state_dict(torch.load(best_model_path, map_location=f'cuda:{local_rank}'))
    # test_model(model, test_dataloader, criterion, local_rank, os.path.join('experiments', args.exp_name))

    # Clean up
    dist.destroy_process_group()


def main():
    """Main function to parse arguments and start training."""
    args = parse_args()

    # Set the master address and port for DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Set visible GPUs
    gpu_list = args.gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list

    # Count number of GPUs to use
    world_size = len(gpu_list.split(','))

    # Create experiment directory
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Start distributed training
    mp.spawn(train_and_evaluate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()