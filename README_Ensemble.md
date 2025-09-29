# Ensemble Learning with Transfer Learning: ResNet50 + XceptionNet

## 📋 Overview

This project implements a comprehensive **Ensemble Learning** approach combined with **Transfer Learning** for medical image classification (Cancer vs Benign). The ensemble combines two powerful deep learning architectures:

- **ResNet50**: Residual Network with 50 layers
- **XceptionNet**: Extreme Inception network with depthwise separable convolutions

## 🎯 Methodology

### 1. Transfer Learning Strategy

**Pre-trained Models:**
- Both ResNet50 and XceptionNet are initialized with ImageNet pre-trained weights
- Early layers (70%) are frozen to preserve learned low-level features
- Final layers are fine-tuned on the cancer detection task

**Benefits:**
- Faster training convergence
- Better performance with limited data
- Leverages learned feature representations from ImageNet

### 2. Ensemble Learning Methods

The implementation provides **four different ensemble methods**:

#### A. **Simple Averaging** (`ensemble_method='average'`)
- Combines predictions by simple arithmetic mean
- Formula: `output = (ResNet50_output + Xception_output) / 2`
- **Pros**: Simple, stable, good baseline
- **Cons**: Treats both models equally

#### B. **Weighted Combination** (`ensemble_method='weighted'`)
- Learns optimal weights for combining predictions
- Weights are learnable parameters optimized during training
- Formula: `output = w1 * ResNet50_output + w2 * Xception_output`
- **Pros**: Adapts to model strengths, often better than averaging
- **Cons**: Slightly more complex, risk of overfitting weights

#### C. **Meta-Learner** (`ensemble_method='meta_learner'`)
- Uses a small neural network to combine predictions
- Concatenates outputs from both models as input to meta-classifier
- **Architecture**: `[ResNet50_out, Xception_out] → Dense(64) → Dense(32) → Dense(2)`
- **Pros**: Can learn complex combination patterns
- **Cons**: More parameters, requires more data

#### D. **Feature Fusion** (`ensemble_method='feature_fusion'`)
- Combines features before final classification
- Extracts features from both models and concatenates them
- **Architecture**: `[ResNet50_features, Xception_features] → Fusion_Classifier`
- **Pros**: Richest feature representation, often best performance
- **Cons**: Highest computational cost, most parameters

### 3. Architecture Details

```
Input Image (224×224×3)
├── ResNet50 Branch
│   ├── Pre-trained backbone (frozen early layers)
│   ├── Feature extraction: 2048-dim
│   └── Classification head: 2048 → 2
└── XceptionNet Branch
    ├── Pre-trained backbone (frozen early layers)
    ├── Feature extraction: 2048-dim
    └── Classification head: 2048 → 2

Ensemble Combination:
├── Average: (ResNet50 + Xception) / 2
├── Weighted: w1*ResNet50 + w2*Xception
├── Meta-learner: MLP([ResNet50, Xception])
└── Feature Fusion: MLP([ResNet50_features, Xception_features])
```

## 📁 File Structure

```
├── ensemble_model.py          # Core ensemble model implementation
├── train_ensemble.py          # Training pipeline
├── predict_ensemble.py        # Prediction and visualization
├── utils.py                   # Utility functions
├── models/                    # Trained model checkpoints
│   ├── ensemble_average_best.pth
│   ├── ensemble_weighted_best.pth
│   ├── ensemble_meta_learner_best.pth
│   └── ensemble_feature_fusion_best.pth
└── data/                      # Dataset
    ├── train/
    │   ├── Benign/
    │   └── Cancer/
    └── val/
        ├── Benign/
        └── Cancer/
```

## 🚀 Usage

### 1. Training an Ensemble Model

```python
from train_ensemble import EnsembleTrainingPipeline

# Single ensemble method
pipeline = EnsembleTrainingPipeline(
    data_dir="/path/to/data",
    ensemble_method='average',  # 'average', 'weighted', 'meta_learner', 'feature_fusion'
    batch_size=16,
    learning_rate=0.0001,
    num_epochs=25
)

# Train and evaluate
best_model_path = pipeline.train()
pipeline.evaluate_model(best_model_path)
pipeline.plot_training_history()
```

### 2. Comparing All Ensemble Methods

```python
from train_ensemble import compare_ensemble_methods

results = compare_ensemble_methods(
    data_dir="/path/to/data",
    methods=['average', 'weighted', 'meta_learner', 'feature_fusion']
)
```

### 3. Making Predictions

```python
from predict_ensemble import EnsemblePredictor

# Load trained model
predictor = EnsemblePredictor(
    model_path="models/ensemble_average_best.pth",
    ensemble_method='average'
)

# Single image prediction
result = predictor.visualize_prediction("path/to/image.png")

# Compare individual model predictions
comparison = predictor.compare_individual_models("path/to/image.png")

# Batch prediction
results = predictor.predict_batch(["img1.png", "img2.png", "img3.png"])
```

## 📊 Expected Performance

Based on ensemble learning principles, expected performance improvements:

| Method | Expected Accuracy | Training Time | Inference Time | Parameters |
|--------|------------------|---------------|----------------|------------|
| ResNet50 (baseline) | ~85-90% | 1x | 1x | ~25M |
| XceptionNet (baseline) | ~87-92% | 1x | 1x | ~23M |
| **Average Ensemble** | ~90-94% | 1.2x | 2x | ~48M |
| **Weighted Ensemble** | ~91-95% | 1.3x | 2x | ~48M |
| **Meta-learner** | ~92-96% | 1.4x | 2.1x | ~48M + meta |
| **Feature Fusion** | ~93-97% | 1.5x | 2.2x | ~50M |

## 🔬 Key Advantages

### 1. **Improved Robustness**
- Reduces overfitting by combining different architectures
- More stable predictions across different data distributions
- Less sensitive to individual model failures

### 2. **Better Generalization**
- ResNet50: Strong spatial feature learning with residual connections
- XceptionNet: Efficient feature extraction with depthwise separations
- Combined: Leverages strengths of both architectures

### 3. **Enhanced Performance**
- Typically 3-5% accuracy improvement over single models
- Better handling of edge cases and difficult samples
- Improved confidence calibration

### 4. **Flexibility**
- Multiple ensemble strategies to choose from
- Easy to add more models to the ensemble
- Configurable ensemble weights and methods

## ⚙️ Training Configuration

### Recommended Hyperparameters:

```python
# For ensemble training
BATCH_SIZE = 16          # Smaller due to memory constraints
LEARNING_RATE = 0.0001   # Lower LR for stable ensemble training
WEIGHT_DECAY = 0.01      # L2 regularization
EPOCHS = 25-50           # Depends on dataset size

# Data augmentation
HORIZONTAL_FLIP = 0.5    # Medical images can be flipped
VERTICAL_FLIP = 0.3      # Less common but useful
ROTATION = 30            # Conservative rotation for medical images
COLOR_JITTER = 0.2       # Brightness/contrast variations
```

### Learning Rate Scheduling:
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- **Patience**: 5 epochs
- **Factor**: 0.5 (halves the learning rate)

## 📈 Evaluation Metrics

The training pipeline provides comprehensive evaluation:

1. **Accuracy**: Overall classification accuracy
2. **Precision/Recall**: Per-class performance
3. **F1-Score**: Harmonic mean of precision and recall
4. **Confusion Matrix**: Detailed classification breakdown
5. **ROC Curves**: Performance across different thresholds
6. **Training Curves**: Loss and accuracy over epochs

## 🛠️ Installation

```bash
# Install required packages
pip install torch torchvision timm
pip install grad-cam matplotlib seaborn
pip install scikit-learn pandas numpy
pip install pillow opencv-python
```

## 🎯 Future Enhancements

1. **Additional Models**: Add EfficientNet, Vision Transformer
2. **Advanced Ensembles**: Bayesian Model Averaging, Stacking
3. **Uncertainty Quantification**: Monte Carlo Dropout, Deep Ensembles
4. **Model Distillation**: Create lightweight ensemble student models
5. **AutoML Integration**: Automated ensemble architecture search

## 📚 References

1. **ResNet**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
2. **Xception**: "Xception: Deep Learning with Depthwise Separable Convolutions" (Chollet, 2017)
3. **Ensemble Methods**: "Ensemble Methods in Machine Learning" (Dietterich, 2000)
4. **Transfer Learning**: "How transferable are features in deep neural networks?" (Yosinski et al., 2014)

---

## 🚀 Quick Start

```bash
# 1. Train ensemble model
python train_ensemble.py

# 2. Make predictions
python predict_ensemble.py

# 3. Compare methods
python -c "from train_ensemble import compare_ensemble_methods; compare_ensemble_methods('data')"
```

This ensemble approach provides a robust, high-performance solution for medical image classification tasks!