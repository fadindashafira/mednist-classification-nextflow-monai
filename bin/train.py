#!/usr/bin/env python

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import numpy as np
from torch.utils.data import DataLoader
from model import get_model

def main():
    parser = argparse.ArgumentParser(description='Train a model for MedNIST classification')
    parser.add_argument('--train_data', required=True, help='Path to training data file')
    parser.add_argument('--val_data', required=True, help='Path to validation data file')
    parser.add_argument('--model_type', default='simple_cnn', help='Model type: simple_cnn, densenet, resnet')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--output_dir', default='./', help='Output directory')
    args = parser.parse_args()
    
    # Check if CUDA is requested but not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Load datasets
    train_data = torch.load(args.train_data, weights_only=False)
    val_data = torch.load(args.val_data, weights_only=False)
    
    train_ds = train_data['dataset']
    val_ds = val_data['dataset']
    num_classes = train_data['num_classes']
    class_names = train_data['class_names']
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize model
    model = get_model(args.model_type, num_classes).to(device)
    print(f"Initialized {args.model_type} model with {num_classes} output classes")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training metrics
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    # Training loop
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f'Epoch: {epoch+1}/{args.epochs}, Batch: {batch_idx+1}/{len(train_loader)}, '
                      f'Loss: {train_loss/(batch_idx+1):.4f}, '
                      f'Acc: {100.*train_correct/train_total:.2f}%')
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f'Epoch: {epoch+1}/{args.epochs}, Time: {epoch_time:.2f}s, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > metrics['best_val_acc']:
            metrics['best_val_acc'] = val_acc
            metrics['best_epoch'] = epoch + 1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
            print(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
    
    # Training summary
    total_time = time.time() - start_time
    metrics['total_time'] = total_time
    metrics['model_type'] = args.model_type
    metrics['epochs'] = args.epochs
    metrics['batch_size'] = args.batch_size
    metrics['learning_rate'] = args.learning_rate
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best model saved at epoch {metrics['best_epoch']} with validation accuracy: {metrics['best_val_acc']:.2f}%")

if __name__ == "__main__":
    main()