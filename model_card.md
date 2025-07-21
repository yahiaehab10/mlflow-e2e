
# Chest CT Scan Classifier Model

## Model Overview
- **Model Type**: Transfer Learning based on VGG16
- **Task**: Binary Classification (Adenocarcinoma vs Normal)
- **Framework**: TensorFlow/Keras
- **Total Parameters**: 14,764,866

## Performance Metrics
- **True Test Accuracy**: 97.01%
- **True Test Loss**: 0.1572
- **Test Set Size**: 67 images

## Model Architecture
Based on VGG16 with custom classification head for binary classification.

## Usage
```python
model = tf.keras.models.load_model("artifacts/training/model.h5")
```

## Training Configuration
{'AUGMENTATION': True, 'IMAGE_SIZE': [224, 224, 3], 'BATCH_SIZE': 16, 'INCLUDE_TOP': False, 'EPOCHS': 1, 'CLASSES': 2, 'WEIGHTS': 'imagenet', 'LEARNING_RATE': 0.01}
