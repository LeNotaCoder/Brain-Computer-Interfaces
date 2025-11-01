"""
Configuration matching paper specifications from:
"A transfer learning-based CNN and LSTM hybrid deep learning model to classify motor imagery EEG signals"
(Khademi et al., 2022)
"""

class Config:
    """Configuration matching paper specifications"""
    
    # Data parameters
    FS = 250  # Sampling frequency (Hz)
    N_CHANNELS = 22  # Number of EEG channels
    N_CLASSES = 4  # left hand, right hand, feet, tongue
    
    # Preprocessing
    LOWCUT = 8  # Hz (Mu band start)
    HIGHCUT = 30  # Hz (Beta band end)
    FILTER_ORDER = 5  # Butterworth filter order
    
    # Data augmentation (Section 3.2-III)
    WINDOW_SIZE = 0.8  # seconds (200 samples at 250 Hz)
    N_CROPS = 5  # Number of crops per trial
    
    # CWT parameters
    WAVELET = 'morl'  # Morlet wavelet
    FREQ_RANGE = (8, 30)  # Hz
    N_SCALES = 64
    
    # Model parameters
    LSTM_HIDDEN = 250  # LSTM units
    FC_HIDDEN = 100  # Fully connected layer
    
    # Training parameters (Section 3.4)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Optimizer and scheduler
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 5
    MIN_LR = 1e-7
    
    # Data split
    TEST_SIZE = 0.2  # 20% for testing
    VAL_SIZE = 0.2  # 20% of train for validation
    
    # Channel for single-channel analysis (C3 - motor cortex)
    ANALYSIS_CHANNEL = 'EEG-C3'
    
    # Model types
    MODELS = ['custom', 'resnet50', 'inception']
    
    # Expected results from paper (Table 6 & 7)
    PAPER_RESULTS = {
        'custom': {'accuracy': 86, 'kappa': 0.81},
        'resnet50': {'accuracy': 90, 'kappa': 0.86},
        'inception': {'accuracy': 92, 'kappa': 0.88}
    }"""
Configuration matching paper specifications from:
"A transfer learning-based CNN and LSTM hybrid deep learning model to classify motor imagery EEG signals"
(Khademi et al., 2022)
"""

class Config:
    """Configuration matching paper specifications"""
    
    # Data parameters
    FS = 250  # Sampling frequency (Hz)
    N_CHANNELS = 22  # Number of EEG channels
    N_CLASSES = 4  # left hand, right hand, feet, tongue
    
    # Preprocessing
    LOWCUT = 8  # Hz (Mu band start)
    HIGHCUT = 30  # Hz (Beta band end)
    FILTER_ORDER = 5  # Butterworth filter order
    
    # Data augmentation (Section 3.2-III)
    WINDOW_SIZE = 0.8  # seconds (200 samples at 250 Hz)
    N_CROPS = 5  # Number of crops per trial
    
    # CWT parameters
    WAVELET = 'morl'  # Morlet wavelet
    FREQ_RANGE = (8, 30)  # Hz
    N_SCALES = 64
    
    # Model parameters
    LSTM_HIDDEN = 250  # LSTM units
    FC_HIDDEN = 100  # Fully connected layer
    
    # Training parameters (Section 3.4)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Optimizer and scheduler
    LR_REDUCE_FACTOR = 0.5
    LR_REDUCE_PATIENCE = 5
    MIN_LR = 1e-7
    
    # Data split
    TEST_SIZE = 0.2  # 20% for testing
    VAL_SIZE = 0.2  # 20% of train for validation
    
    # Channel for single-channel analysis (C3 - motor cortex)
    ANALYSIS_CHANNEL = 'EEG-C3'
    
    # Model types
    MODELS = ['custom', 'resnet50', 'inception']
    
    # Expected results from paper (Table 6 & 7)
    PAPER_RESULTS = {
        'custom': {'accuracy': 86, 'kappa': 0.81},
        'resnet50': {'accuracy': 90, 'kappa': 0.86},
        'inception': {'accuracy': 92, 'kappa': 0.88}
    }
