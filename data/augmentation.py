import numpy as np


def crop_augmentation(data, labels, window_size=0.8, n_crops=5, fs=250):
    """
    Non-overlapping crop augmentation (Section 3.2-III)
    Increases dataset by factor of 5
    """
    n_trials, n_channels, n_samples = data.shape
    window_samples = int(window_size * fs)
    
    augmented_data = []
    augmented_labels = []
    
    for i in range(n_trials):
        trial = data[i]
        label = labels[i]
        
        # Non-overlapping crops
        step = (n_samples - window_samples) // (n_crops - 1) if n_crops > 1 else 0
        for j in range(n_crops):
            start_idx = j * step
            end_idx = start_idx + window_samples
            if end_idx <= n_samples:
                crop = trial[:, start_idx:end_idx]
                augmented_data.append(crop)
                augmented_labels.append(label)
    
    return np.array(augmented_data), np.array(augmented_labels)
