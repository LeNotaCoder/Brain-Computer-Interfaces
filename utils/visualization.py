import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(true_labels, pred_labels, model_type, accuracy, kappa, save_path=None):
    """Plot confusion matrix with results"""
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['LH', 'RH', 'Feet', 'Tongue'],
                yticklabels=['LH', 'RH', 'Feet', 'Tongue'])
    plt.title(f'Confusion Matrix - {model_type.upper()}\nAcc: {accuracy:.2f}%, Kappa: {kappa:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
