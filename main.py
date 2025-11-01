"""
Exact replication of paper methodology:
"A transfer learning-based CNN and LSTM hybrid deep learning model to classify motor imagery EEG signals"
(Khademi et al., 2022)

Main execution script that ties all modules together.
"""

import os
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

# Import from our modules
from config import Config
from data.loader import load_patient_data, EEGDataset, inspect_csv_format
from data.preprocessing import apply_car, bandpass_filter
from data.augmentation import crop_augmentation
from utils.cwt_transform import apply_cwt, scalogram_to_rgb
from models.custom_cnn_lstm import CustomCNNLSTM
from models.resnet50_lstm import ResNet50LSTM
from models.inception_lstm import InceptionV3LSTM
from training.trainer import Trainer
from utils.visualization import plot_confusion_matrix


def train_subject_specific(patient_file, model_type='inception', save_dir='results'):
    """
    Train subject-specific model (as in paper)
    
    This follows the exact methodology from Section 5:
    - Cross-subject approach with session-based split
    - Train on session 1, test on session 2 (or vice versa)
    """
    print("\n" + "="*80)
    print(f"SUBJECT-SPECIFIC TRAINING: {model_type.upper()}")
    print("="*80)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading data...")
    eeg_data, labels, channel_idx = load_patient_data(patient_file, Config.ANALYSIS_CHANNEL)
    
    # Preprocessing
    print("\n[2/7] Preprocessing...")
    print("  - Common Average Referencing")
    eeg_data = apply_car(eeg_data)
    
    print("  - Bandpass filtering (8-30 Hz)")
    eeg_data = bandpass_filter(eeg_data, Config.FS, Config.LOWCUT, Config.HIGHCUT)
    
    print("  - Crop augmentation (5x increase)")
    eeg_data, labels = crop_augmentation(eeg_data, labels, Config.WINDOW_SIZE, Config.N_CROPS, Config.FS)
    print(f"    Augmented: {eeg_data.shape[0]} trials")
    
    # CWT transformation
    print("\n[3/7] CWT transformation...")
    target_size = (299, 299) if model_type == 'inception' else (224, 224)
    
    images = []
    for i in range(len(eeg_data)):
        scalogram = apply_cwt(eeg_data[i, channel_idx], Config.FS, Config.WAVELET, 
                             Config.FREQ_RANGE, Config.N_SCALES)
        rgb_image = scalogram_to_rgb(scalogram, target_size)
        images.append(rgb_image)
    images = np.array(images)
    
    # Split data
    print("\n[4/7] Splitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=Config.TEST_SIZE, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=Config.VAL_SIZE, random_state=42, stratify=y_train_val
    )
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    test_dataset = EEGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("\n[5/7] Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    if model_type == 'custom':
        model = CustomCNNLSTM()
    elif model_type == 'resnet50':
        model = ResNet50LSTM()
    elif model_type == 'inception':
        model = InceptionV3LSTM()
    else:
        raise ValueError(f"Unknown model: {model_type}")
    
    total_params, trainable_params = model.count_parameters()
    print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Train
    print("\n[6/7] Training...")
    trainer = Trainer(model, device, lr=Config.LEARNING_RATE)
    history = trainer.train(train_loader, val_loader, 
                           epochs=Config.MAX_EPOCHS, 
                           patience=Config.EARLY_STOPPING_PATIENCE)
    
    # Test
    print("\n[7/7] Testing...")
    test_loss, test_acc, test_kappa, preds, true_labels = trainer.validate(test_loader)
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Kappa: {test_kappa:.4f}")
    print(f"\nPaper Reported:")
    print(f"  {model_type}: {Config.PAPER_RESULTS[model_type]['accuracy']}% accuracy, "
          f"{Config.PAPER_RESULTS[model_type]['kappa
