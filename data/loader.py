import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def inspect_csv_format(csv_path):
    """Inspect CSV file format to understand data structure"""
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*80)
    print("CSV FORMAT INSPECTION")
    print("="*80)
    print(f"\nFile: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    print(f"\nUnique labels: {df['label'].unique()}")
    print(f"Label counts:\n{df['label'].value_counts()}")
    print(f"\nUnique epochs: {len(df['epoch'].unique())}")
    print(f"Samples per epoch (approx): {df.groupby('epoch').size().mean():.0f}")
    print("="*80 + "\n")


def load_patient_data(csv_path, channel_name='EEG-C3'):
    """Load and preprocess single patient data"""
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Get channel index
    eeg_columns = [col for col in df.columns if col.startswith('EEG')]
    if channel_name not in eeg_columns:
        print(f"Warning: {channel_name} not found, using first channel")
        channel_idx = 0
    else:
        channel_idx = eeg_columns.index(channel_name)
    
    # Extract data by epochs
    epochs = df['epoch'].unique()
    n_epochs = len(epochs)
    samples_per_epoch = df[df['epoch'] == epochs[0]].shape[0]
    n_channels = len(eeg_columns)
    
    eeg_data = np.zeros((n_epochs, n_channels, samples_per_epoch))
    labels = np.zeros(n_epochs, dtype=np.int64)
    
    # Check what labels exist in the data
    unique_labels = df['label'].unique()
    print(f"  Unique labels found: {unique_labels}")
    
    # Flexible label mapping - handle all variations
    label_mapping = {
        # Standard format
        'left hand': 0, 'right hand': 1, 'feet': 2, 'tongue': 3,
        # Short format (what's actually in your CSVs)
        'left': 0, 'right': 1, 'foot': 2, 'tongue': 3,
        # Alternative formats
        'left_hand': 0, 'right_hand': 1,
        'LEFT HAND': 0, 'RIGHT HAND': 1, 'FEET': 2, 'TONGUE': 3,
        'LEFT': 0, 'RIGHT': 1, 'FOOT': 2,
        # Numeric (if already encoded)
        0: 0, 1: 1, 2: 2, 3: 3
    }
    
    for idx, epoch_id in enumerate(epochs):
        epoch_df = df[df['epoch'] == epoch_id]
        eeg_data[idx] = epoch_df[eeg_columns].values.T
        
        label_value = epoch_df['label'].iloc[0]
        if label_value in label_mapping:
            labels[idx] = label_mapping[label_value]
        else:
            print(f"  Warning: Unknown label '{label_value}' found, skipping this epoch")
            labels[idx] = -1
    
    # Remove epochs with invalid labels
    valid_mask = labels != -1
    if not valid_mask.all():
        print(f"  Removing {(~valid_mask).sum()} epochs with invalid labels")
        eeg_data = eeg_data[valid_mask]
        labels = labels[valid_mask]
        n_epochs = len(labels)
    
    print(f"  Loaded: {n_epochs} epochs, {n_channels} channels, {samples_per_epoch} samples")
    print(f"  Label distribution: {dict(zip(['left hand', 'right hand', 'feet', 'tongue'], np.bincount(labels)))}")
    
    return eeg_data, labels, channel_idx


class EEGDataset(Dataset):
    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
