import torch
import torch.nn as nn

from timm import create_model

from utils import DEVICE
from dataloader import get_val_transforms


class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.backbone = create_model(
            model_name, pretrained=True, num_classes=2)

        # remove the final classification layer
        self.backbone.reset_classifier(0)

        self.cancer_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

        self.level_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        # Define the forward pass
        features = self.backbone(x)
        cancer_output = self.cancer_classifier(features)
        level_output = self.level_classifier(features)
        return cancer_output, level_output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            cancer_output, level_output = self.forward(x)
            cancer_probs = torch.nn.functional.softmax(cancer_output, dim=1)
            level_probs = torch.nn.functional.softmax(level_output, dim=1)
            return cancer_probs, level_probs

    def load_pretrained_weights(self, weight_path):
        """Load pretrained weights from `weight_path`.

        This helper handles common checkpoint formats:
        - raw state_dict saved with `torch.save(model.state_dict())`
        - checkpoint dicts with keys like `'state_dict'` or `'model_state_dict'`
        It also strips the `module.` prefix that appears when models were saved
        from `nn.DataParallel`.

        By default the load is non-strict where possible and the function prints
        any missing or unexpected keys to help debugging.
        """
        # load checkpoint onto configured device
        checkpoint = torch.load(weight_path, map_location=DEVICE)

        # extract state_dict if checkpoint is a dict with common keys
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                # assume the dict is already a state_dict-like mapping
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # strip `module.` prefix if present (from DataParallel) when target model
        # does not use DataParallel
        new_state = {}
        own_state = self.state_dict()
        for k, v in state_dict.items():
            new_k = k
            if k.startswith('module.') and new_k[len('module.'):] in own_state:
                new_k = new_k[len('module.'):]
            new_state[new_k] = v

        # try to load, prefer non-strict to avoid failures due to small mismatches
        try:
            load_result = self.load_state_dict(new_state, strict=False)
            missing = getattr(load_result, 'missing_keys', None)
            unexpected = getattr(load_result, 'unexpected_keys', None)
            if missing:
                print(
                    f"Warning: missing keys when loading pretrained weights: {missing}")
            if unexpected:
                print(
                    f"Warning: unexpected keys when loading pretrained weights: {unexpected}")
        except Exception as e:
            # fallback: raise with context
            raise RuntimeError(
                f"Failed to load pretrained weights from {weight_path}: {e}")

    def model_predict(self, img_pil):
        """Make prediction on the image"""

        # val transform
        val_transform = get_val_transforms()

        # transform image
        image_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            return predicted_class.item(), confidence.item(), probabilities[0].cpu().numpy()
