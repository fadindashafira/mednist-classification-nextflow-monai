# MedNIST Classification with Nextflow and MONAI

A pipeline for medical image classification using the MedNIST dataset, MONAI framework, and Nextflow workflow management system.

## Overview

This project implements a complete workflow for training and evaluating a medical image classification model using the MedNIST dataset. It is structured as a Nextflow pipeline to enable scalable, reproducible execution across different computing environments.

The pipeline consists of three main stages:
1. **Data Preparation**: Download and preprocess the MedNIST dataset
2. **Model Training**: Train a deep learning model on the preprocessed data
3. **Model Evaluation**: Evaluate the trained model on a test set

## Features

- **Automatic Dataset Management**: Automatic downloading and extraction of the MedNIST dataset
- **Simple CNN Architecture**: Lightweight convolutional neural network for medical image classification
- **Reproducible Workflow**: Consistent execution across different environments
- **Comprehensive Evaluation**: Generate detailed metrics, confusion matrix, and ROC curve

## Requirements

- [Nextflow](https://www.nextflow.io/) (21.10.0 or later)
- [Python](https://www.python.org/) (3.8 or later)
- [PyTorch](https://pytorch.org/) (1.9 or later)
- [MONAI](https://monai.io/) (0.9 or later)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)

## Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/fadindashafira/mednist-classification-nextflow-monai.git
cd mednist-classification-nextflow-monai
```

2. Install required dependencies:
```bash
pip install torch monai scikit-learn matplotlib pandas numpy tqdm
```

### Running the Pipeline

Run the pipeline with default parameters:
```bash
nextflow run main.nf
```

This will automatically:
1. Download the MedNIST dataset (~167MB) if not already present
2. Preprocess the data
3. Train a Simple CNN model
4. Evaluate the model and generate metrics

### Customizing Parameters

You can customize parameters by modifying `conf/params.config` or by passing parameters directly:

```bash
nextflow run main.nf --epochs 10 --batch_size 32 --learning_rate 0.001
```

## Project Structure

```
mednist-nextflow/
├── main.nf                 # Main Nextflow workflow script
├── nextflow.config         # Nextflow configuration
├── bin/                    # Python scripts for each step
│   ├── data.py             # Data preparation
│   ├── model.py            # Model definition
│   ├── train.py            # Model training
│   └── evaluate.py         # Model evaluation
├── modules/                # Nextflow modules
│   ├── prepare_data.nf     # Data preparation process
│   ├── train_model.nf      # Model training process
│   └── evaluate_model.nf   # Model evaluation process
├── conf/                   # Configuration files
│   ├── base.config         # Base configuration
│   ├── params.config       # Pipeline parameters
│   └── resources.config    # Resource allocation
```

## Outputs

Results are saved in the output directory (default: `./results`):

- `data/`: Preprocessed datasets
- `model/`: Trained model and training metrics
- `evaluation/`: Evaluation metrics, confusion matrix, and ROC curve
- Execution reports: Timeline, trace, and summary in HTML format

## Dataset

The [MedNIST dataset](https://github.com/Project-MONAI/MONAI-extra-test-data) contains 2D medical images from 6 classes:
- AbdomenCT
- BreastMRI
- CXR (Chest X-Ray)
- ChestCT
- Hand
- HeadCT

The pipeline automatically downloads and preprocesses this dataset.

## Model Architecture

The project uses a simple Convolutional Neural Network (CNN) with:
- 3 convolutional layers with ReLU activation and max pooling
- Flatten layer
- 2 fully connected layers with dropout for regularization

## Acknowledgments

- [MONAI](https://monai.io/) for the medical imaging framework
- [Nextflow](https://www.nextflow.io/) for the workflow management system
- [MedNIST dataset](https://github.com/Project-MONAI/MONAI-extra-test-data) for the medical images
