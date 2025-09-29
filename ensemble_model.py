import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model
from utils import DEVICE, CLASS_NAMES


class EnsembleModel(nn.Module):
    """
    Ensemble model combining ResNet50 and XceptionNet with Transfer Learning
    """

    def __init__(self, num_classes=2, ensemble_method='average', dropout_rate=0.5):
        super(EnsembleModel, self).__init__()

        self.ensemble_method = ensemble_method
        self.num_classes = num_classes

        # Model 1: ResNet50 with Transfer Learning
        self.resnet50 = create_model(
            'resnet50', pretrained=True, num_classes=num_classes)

        # Model 2: XceptionNet with Transfer Learning
        self.xception = create_model(
            'xception', pretrained=True, num_classes=num_classes)

        # Freeze early layers for transfer learning (optional)
        self._freeze_early_layers()

        # Ensemble combination layers
        if ensemble_method == 'weighted':
            self.weight_resnet = nn.Parameter(torch.tensor(0.5))
            self.weight_xception = nn.Parameter(torch.tensor(0.5))

        elif ensemble_method == 'meta_learner':
            # Meta-learner approach
            self.meta_classifier = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(32, num_classes)
            )

        elif ensemble_method == 'feature_fusion':
            # Feature fusion before final classification
            # Get feature dimensions from both models
            resnet_features = 2048  # ResNet50 feature dimension
            xception_features = 2048  # Xception feature dimension

            # Remove final classification layers
            self.resnet50.reset_classifier(0)  # Remove classifier
            self.xception.reset_classifier(0)   # Remove classifier

            # New fusion classifier
            self.fusion_classifier = nn.Sequential(
                nn.Linear(resnet_features + xception_features, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )

    def _freeze_early_layers(self, freeze_ratio=0.7):
        """
        Freeze early layers for transfer learning
        """
        # Freeze early layers in ResNet50
        resnet_layers = list(self.resnet50.children())
        freeze_until_resnet = int(len(resnet_layers) * freeze_ratio)

        for i, layer in enumerate(resnet_layers):
            if i < freeze_until_resnet:
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze early layers in Xception
        xception_layers = list(self.xception.children())
        freeze_until_xception = int(len(xception_layers) * freeze_ratio)

        for i, layer in enumerate(xception_layers):
            if i < freeze_until_xception:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        if self.ensemble_method == 'average':
            # Simple averaging
            resnet_out = self.resnet50(x)
            xception_out = self.xception(x)
            return (resnet_out + xception_out) / 2

        elif self.ensemble_method == 'weighted':
            # Learned weighted combination
            resnet_out = self.resnet50(x)
            xception_out = self.xception(x)

            # Normalize weights
            w1 = torch.sigmoid(self.weight_resnet)
            w2 = torch.sigmoid(self.weight_xception)
            total_weight = w1 + w2
            w1 = w1 / total_weight
            w2 = w2 / total_weight

            return w1 * resnet_out + w2 * xception_out

        elif self.ensemble_method == 'meta_learner':
            # Meta-learner approach
            resnet_out = self.resnet50(x)
            xception_out = self.xception(x)

            # Concatenate predictions
            combined = torch.cat([resnet_out, xception_out], dim=1)
            return self.meta_classifier(combined)

        elif self.ensemble_method == 'feature_fusion':
            # Feature fusion approach
            # Features before classification
            resnet_features = self.resnet50(x)
            # Features before classification
            xception_features = self.xception(x)

            # Concatenate features
            fused_features = torch.cat(
                [resnet_features, xception_features], dim=1)
            return self.fusion_classifier(fused_features)

    def get_individual_predictions(self, x):
        """
        Get predictions from individual models (useful for analysis)
        """
        with torch.no_grad():
            if self.ensemble_method == 'feature_fusion':
                # Need to temporarily restore classifiers for individual predictions
                resnet_temp = create_model(
                    'resnet50', pretrained=False, num_classes=self.num_classes)
                xception_temp = create_model(
                    'xception', pretrained=False, num_classes=self.num_classes)
            else:
                resnet_out = self.resnet50(x)
                xception_out = self.xception(x)
                return F.softmax(resnet_out, dim=1), F.softmax(xception_out, dim=1)


class EnsembleTrainer:
    """
    Training class for the ensemble model
    """

    def __init__(self, model, criterion, optimizer, device=DEVICE):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, train_loader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.6f}')

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)

        return avg_loss, accuracy

    def validate(self, val_loader):
        """
        Validate the model
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100. * correct / total
        avg_loss = val_loss / len(val_loader)

        return avg_loss, accuracy


def create_ensemble_model(ensemble_method='average', num_classes=2):
    """
    Factory function to create ensemble model

    Args:
        ensemble_method: 'average', 'weighted', 'meta_learner', 'feature_fusion'
        num_classes: Number of output classes

    Returns:
        EnsembleModel instance
    """
    model = EnsembleModel(
        num_classes=num_classes,
        ensemble_method=ensemble_method,
        dropout_rate=0.5
    ).to(DEVICE)

    return model


def get_ensemble_transforms():
    """
    Get transforms that work for both ResNet50 (224x224) and Xception (299x299)
    We'll use 224x224 as common size for efficiency
    """
    from torchvision import transforms
    from utils import PadToMaxSize

    train_transform = transforms.Compose([
        PadToMaxSize(),
        transforms.Resize((224, 224)),  # Common size for both models
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        # Reduced rotation for stability
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        PadToMaxSize(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


if __name__ == "__main__":
    # Example usage
    print("Available ensemble methods:")
    print("1. 'average' - Simple averaging of predictions")
    print("2. 'weighted' - Learned weighted combination")
    print("3. 'meta_learner' - Meta-classifier on combined predictions")
    print("4. 'feature_fusion' - Feature-level fusion")

    # Create model
    model = create_ensemble_model(ensemble_method='average')
    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
