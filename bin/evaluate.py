#!/usr/bin/env python

import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from model import get_model

def plot_confusion_matrix(cm, class_names, output_file):
    """
    Plot confusion matrix as a heatmap and save to file
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_file)
    plt.close()

def plot_roc_curve(y_true, y_score, class_names, output_file):
    """
    Plot ROC curves for multi-class classification and save to file
    """
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on MedNIST test data')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--test_data', required=True, help='Path to test data file')
    parser.add_argument('--model_type', default='simple_cnn', help='Model type: simple_cnn, densenet, resnet')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for evaluation (cuda/cpu)')
    parser.add_argument('--output_dir', default='./', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    args = parser.parse_args()
    
    # Check if CUDA is requested but not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Using CPU instead.")
        args.device = 'cpu'
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    test_data = torch.load(args.test_data, weights_only=False)
    test_ds = test_data['dataset']
    num_classes = test_data['num_classes']
    class_names = test_data['class_names']
    
    # Create data loader
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    model = get_model(args.model_type, num_classes).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Loaded model from {args.model}")
    
    # Initialize variables for evaluation
    all_preds = []
    all_labels = []
    all_scores = []  # For ROC curve
    
    # Evaluate the model
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            
            outputs = model(inputs)
            scores = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names, 
                                  output_dict=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, os.path.join(args.output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plot_roc_curve(all_labels, all_scores, class_names, os.path.join(args.output_dir, 'roc_curve.png'))
    
    # Create evaluation metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'class_report': report,
        'confusion_matrix': cm.tolist(),
        'model_type': args.model_type
    }
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()