#!/usr/bin/env python

import os
import torch
import argparse
import numpy as np
import urllib.request
import tarfile
import hashlib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandRotated,
    RandFlipd,
    ToTensord,
)
from monai.data import CacheDataset, Dataset

def calculate_md5(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_and_extract(url, filepath, output_dir, md5=None):
    """
    Download and extract a tar.gz file
    """
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the file if it doesn't exist or md5 doesn't match
    if not os.path.exists(filepath) or (md5 and calculate_md5(filepath) != md5):
        print(f"Downloading {url} to {filepath}")
        urllib.request.urlretrieve(url, filepath)
        
        # Verify md5
        if md5 and calculate_md5(filepath) != md5:
            raise RuntimeError(f"MD5 mismatch for downloaded file: {filepath}")
    
    # Extract the file
    if tarfile.is_tarfile(filepath):
        print(f"Extracting {filepath} to {output_dir}")
        with tarfile.open(filepath) as tar:
            tar.extractall(path=output_dir)

def ensure_mednist_data(data_dir):
    """
    Ensure MedNIST dataset is available, download if needed
    """
    # Define MedNIST dataset details
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    
    # Get the root directory (parent of data_dir)
    root_dir = os.path.dirname(data_dir)
    
    # Define paths
    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    
    # Check if data directory exists and contains data
    if os.path.exists(data_dir):
        # Verify if directory contains expected data
        expected_classes = ['AbdomenCT', 'BreastMRI', 'CXR', 'ChestCT', 'Hand', 'HeadCT']
        existing_classes = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))]
        
        if all(cls in existing_classes for cls in expected_classes):
            print(f"MedNIST dataset found at {data_dir}")
            return
        else:
            print(f"MedNIST dataset at {data_dir} appears incomplete. Downloading fresh copy...")
    else:
        print(f"MedNIST dataset not found at {data_dir}, downloading...")
    
    # Download and extract the dataset
    download_and_extract(resource, compressed_file, root_dir, md5)

def main():
    parser = argparse.ArgumentParser(description='Prepare MedNIST data for classification')
    parser.add_argument('--data_dir', required=True, help='Directory containing MedNIST dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed data')
    parser.add_argument('--cache_rate', type=float, default=1.0, help='Cache rate for dataset')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--test_frac', type=float, default=0.2, help='Fraction of data for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--download', action='store_true', help='Download dataset if not available')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Ensure MedNIST data is available
    if args.download:
        ensure_mednist_data(args.data_dir)
    
    # Set up data transforms - with dictionary transforms
    train_transforms = Compose([
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(64, 64)),
        RandRotated(keys=["img"], range_x=15, prob=0.5, keep_size=True),
        RandFlipd(keys=["img"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["img"], spatial_axis=1, prob=0.5),
        ToTensord(keys=["img"]),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"]),
        Resized(keys=["img"], spatial_size=(64, 64)),
        ToTensord(keys=["img"]),
    ])
    
    # Get class names and create file lists
    class_names = sorted([x for x in os.listdir(args.data_dir) 
                         if os.path.isdir(os.path.join(args.data_dir, x))])
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    
    # Create data lists for each class
    all_files = []
    for i, cls in enumerate(class_names):
        cls_dir = os.path.join(args.data_dir, cls)
        cls_files = [{"img": os.path.join(cls_dir, x), "label": i} 
                    for x in os.listdir(cls_dir) 
                    if os.path.isfile(os.path.join(cls_dir, x))]
        all_files.extend(cls_files)
    
    # Shuffle files
    np.random.shuffle(all_files)
    
    # Split into train, validation, and test sets
    val_size = int(len(all_files) * args.val_frac)
    test_size = int(len(all_files) * args.test_frac)
    train_size = len(all_files) - val_size - test_size
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    print(f"Data split: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}")
    
    # Using Dataset instead of CacheDataset to avoid caching issues
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save datasets
    torch.save({
        'dataset': train_ds,
        'class_names': class_names,
        'num_classes': num_classes
    }, os.path.join(args.output_dir, 'train_data.pt'))
    
    torch.save({
        'dataset': val_ds,
        'class_names': class_names,
        'num_classes': num_classes
    }, os.path.join(args.output_dir, 'val_data.pt'))
    
    torch.save({
        'dataset': test_ds,
        'class_names': class_names,
        'num_classes': num_classes
    }, os.path.join(args.output_dir, 'test_data.pt'))
    
    print(f"Data preparation complete. Files saved to {args.output_dir}")

if __name__ == "__main__":
    main()