import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

from ensemble_model import create_ensemble_model, EnsembleTrainer, get_ensemble_transforms
from utils import DEVICE, CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an ensemble deep learning model')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing the training data (default: data)')
    parser.add_argument('--ensemble-method', type=str, default='average',
                        choices=['average', 'weighted', 'meta_learner'],
                        help='Ensemble method to use (default: average)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate for optimizer (default: 0.0001)')
    return parser.parse_args()

# Parse command line arguments
args = parse_args()

class EnsembleTrainingPipeline:
    """
    Complete training pipeline for ensemble models
    """

    def __init__(self, data_dir, ensemble_method='average', batch_size=32,
                 learning_rate=0.001, num_epochs=50):

        self.data_dir = data_dir
        self.ensemble_method = ensemble_method
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = DEVICE

        # Create model
        self.model = create_ensemble_model(ensemble_method=ensemble_method)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=learning_rate, weight_decay=0.01)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )

        # Trainer
        self.trainer = EnsembleTrainer(
            self.model, self.criterion, self.optimizer)

        # Data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()

        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def _create_data_loaders(self):
        """
        Create data loaders for training and validation
        """
        train_transform, val_transform = get_ensemble_transforms()

        # Training dataset
        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'train'),
            transform=train_transform
        )

        # Validation dataset
        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, 'val'),
            transform=val_transform
        )

        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Classes: {train_dataset.classes}")

        return train_loader, val_loader

    def train(self):
        """
        Complete training loop
        """
        print(
            f"Starting training with {self.ensemble_method} ensemble method...")
        print(f"Device: {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        best_val_acc = 0
        best_model_path = f'models/ensemble_{self.ensemble_method}_best.pth'

        # Create models directory
        os.makedirs('models', exist_ok=True)

        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 50)

            # Training phase
            train_loss, train_acc = self.trainer.train_epoch(self.train_loader)

            # Validation phase
            val_loss, val_acc = self.trainer.validate(self.val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Print epoch results
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'ensemble_method': self.ensemble_method
                }, best_model_path)
                print(f'New best model saved! Val Acc: {best_val_acc:.2f}%')

        # Save last model
        last_model_path = f'models/ensemble_{self.ensemble_method}_{datetime.now().strftime("%Y%m%d")}_last.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'ensemble_method': self.ensemble_method
        }, last_model_path)

        # Update best model path with date
        dated_best_model_path = f'models/ensemble_{self.ensemble_method}_{datetime.now().strftime("%Y%m%d")}_best.pth'
        if os.path.exists(best_model_path):
            os.rename(best_model_path, dated_best_model_path)
            best_model_path = dated_best_model_path

        # Log training completion with specified format
        print(f'\nTraining completed for {self.num_epochs} epochs')
        print(
            f'Last model: ensemble_{self.ensemble_method}_{datetime.now().strftime("%Y%m%d")}_last.pth')
        print(
            f'Best model: ensemble_{self.ensemble_method}_{datetime.now().strftime("%Y%m%d")}_best.pth with accuracy: {best_val_acc/100:.4f}')

        # Convert accuracy percentages to decimals for logging
        train_accs_decimal = [acc/100 for acc in self.history['train_acc']]
        val_accs_decimal = [acc/100 for acc in self.history['val_acc']]

        print(f'train_accuracies: {train_accs_decimal}')
        print(f'val_accuracies: {val_accs_decimal}')
        print(f'train_losses: {self.history["train_loss"]}')
        print(f'val_losses: {self.history["val_loss"]}')

        return best_model_path

    def evaluate_model(self, model_path):
        """
        Comprehensive model evaluation
        """
        # Load best model
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)

                all_preds.extend(output.argmax(dim=1).cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels,
              all_preds, target_names=CLASS_NAMES))

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(
            f'Confusion Matrix - {self.ensemble_method.title()} Ensemble')
        plt.colorbar()
        tick_marks = np.arange(len(CLASS_NAMES))
        plt.xticks(tick_marks, CLASS_NAMES)
        plt.yticks(tick_marks, CLASS_NAMES)

        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'ensemble_{self.ensemble_method}_confusion_matrix.png')
        plt.show()

        return all_probs, all_preds, all_labels

    def plot_training_history(self):
        """
        Plot training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(self.history['train_loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title(f'{self.ensemble_method.title()} Ensemble - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Training Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title(f'{self.ensemble_method.title()} Ensemble - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(f'ensemble_{self.ensemble_method}_training_history.png')
        plt.show()

        # save history to a file with specified format
        history_path = f'models/ensemble_{self.ensemble_method}_{datetime.now().strftime("%Y%m%d")}_log.txt'
        with open(history_path, 'w') as f:
            f.write(
                f"Training completed for {len(self.history['train_loss'])} epochs\n")
            f.write(
                f"Last model: ensemble_{self.ensemble_method}_{datetime.now().strftime('%Y%m%d')}_last.pth\n")

            best_val_acc = max(self.history['val_acc'])
            f.write(
                f"Best model: ensemble_{self.ensemble_method}_{datetime.now().strftime('%Y%m%d')}_best.pth with accuracy: {best_val_acc/100:.4f}\n")

            # Convert accuracy percentages to decimals for logging
            train_accs_decimal = [acc/100 for acc in self.history['train_acc']]
            val_accs_decimal = [acc/100 for acc in self.history['val_acc']]

            f.write(f"train_accuracies: {train_accs_decimal}\n")
            f.write(f"val_accuracies: {val_accs_decimal}\n")
            f.write(f"train_losses: {self.history['train_loss']}\n")
            f.write(f"val_losses: {self.history['val_loss']}\n")


def compare_ensemble_methods(data_dir, methods=['average', 'weighted', 'meta_learner']):
    """
    Compare different ensemble methods
    """
    results = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Training with {method} ensemble method")
        print(f"{'='*60}")

        # Create and train pipeline
        pipeline = EnsembleTrainingPipeline(
            data_dir=data_dir,
            ensemble_method=method,
            batch_size=16,  # Smaller batch size for ensemble
            learning_rate=0.0001,  # Lower learning rate for ensemble
            num_epochs=30
        )

        # Train model
        best_model_path = pipeline.train()

        # Evaluate model
        probs, preds, labels = pipeline.evaluate_model(best_model_path)

        # Plot training history
        pipeline.plot_training_history()

        # Store results
        val_acc = max(pipeline.history['val_acc'])
        results[method] = {
            'val_accuracy': val_acc,
            'model_path': best_model_path,
            'history': pipeline.history
        }

    # Compare results
    print(f"\n{'='*60}")
    print("ENSEMBLE METHOD COMPARISON")
    print(f"{'='*60}")

    for method, result in results.items():
        print(f"{method.title()}: {result['val_accuracy']:.2f}%")

    # Find best method
    best_method = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
    print(f"\nBest ensemble method: {best_method.title()} "
          f"({results[best_method]['val_accuracy']:.2f}%)")

    return results


def log_training_results(model_name, num_epochs, train_accuracies, val_accuracies,
                         train_losses, val_losses, best_accuracy, date_str=None):
    """
    Log training results in the specified format

    Args:
        model_name: Name of the model (e.g., 'resnet50', 'xception')
        num_epochs: Number of training epochs
        train_accuracies: List of training accuracies (as decimals, not percentages)
        val_accuracies: List of validation accuracies (as decimals, not percentages)
        train_losses: List of training losses
        val_losses: List of validation losses
        best_accuracy: Best validation accuracy (as decimal)
        date_str: Date string (YYYYMMDD format), if None uses current date
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    # Print to console
    print(f"Training completed for {num_epochs} epochs")
    print(f"Last model: {model_name}_{date_str}_last.pth")
    print(
        f"Best model: {model_name}_{date_str}_best.pth with accuracy: {best_accuracy:.4f}")
    print(f"train_accuracies: {train_accuracies}")
    print(f"val_accuracies: {val_accuracies}")
    print(f"train_losses: {train_losses}")
    print(f"val_losses: {val_losses}")

    # Save to file
    log_path = f"{model_name}_{date_str}_training_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Training completed for {num_epochs} epochs\n")
        f.write(f"Last model: {model_name}_{date_str}_last.pth\n")
        f.write(
            f"Best model: {model_name}_{date_str}_best.pth with accuracy: {best_accuracy:.4f}\n")
        f.write(f"train_accuracies: {train_accuracies}\n")
        f.write(f"val_accuracies: {val_accuracies}\n")
        f.write(f"train_losses: {train_losses}\n")
        f.write(f"val_losses: {val_losses}\n")

    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":

    # Single ensemble training
    pipeline = EnsembleTrainingPipeline(
        data_dir=args.data_dir,
        ensemble_method=args.ensemble_method,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs
    )

    # Train and evaluate
    best_model_path = pipeline.train()
    pipeline.evaluate_model(best_model_path)
    pipeline.plot_training_history()

    # Uncomment to compare all methods (will take longer)
    # results = compare_ensemble_methods(data_dir)

    # Example usage of standalone logging function:
    # log_training_results(
    #     model_name="resnet50",
    #     num_epochs=30,
    #     train_accuracies=[0.6989, 0.7826, 0.8051, ...],  # Your training accuracies
    #     val_accuracies=[0.6806, 0.6806, 0.7709, ...],     # Your validation accuracies
    #     train_losses=[0.5721, 0.4753, 0.4314, ...],       # Your training losses
    #     val_losses=[0.7194, 1.1865, 0.4997, ...],         # Your validation losses
    #     best_accuracy=0.9387,                              # Best validation accuracy
    #     date_str="20250927"                                # Optional date string
    # )
