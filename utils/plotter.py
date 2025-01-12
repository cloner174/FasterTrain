#
import matplotlib.pyplot as plt


def plot_metrics(metrics, plot_path):
    
    epochs = metrics['epoch']
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train/box_loss'], label='Train Box Loss')
    plt.plot(epochs, metrics['train/cls_loss'], label='Train Cls Loss')
    plt.plot(epochs, metrics['train/dfl_loss'], label='Train DFL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['val/box_loss'], label='Val Box Loss')
    plt.plot(epochs, metrics['val/cls_loss'], label='Val Cls Loss')
    plt.plot(epochs, metrics['val/dfl_loss'], label='Val DFL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['metrics/precision'], label='Precision')
    plt.plot(epochs, metrics['metrics/recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Precision & Recall')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, metrics['metrics/mAP50'], label='mAP50')
    plt.plot(epochs, metrics['metrics/mAP50-95'], label='mAP50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


#cloner174