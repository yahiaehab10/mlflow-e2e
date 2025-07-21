
# Model Deployment Guide

## Loading Models in Production:

### By Latest Version:
```python
import mlflow
model = mlflow.keras.load_model("models://ChestCTScanClassifier/latest")
```

### By Specific Version:
```python
model = mlflow.keras.load_model("models://ChestCTScanClassifier/1")
```

### By Stage:
```python
# Load staging model for testing
staging_model = mlflow.keras.load_model("models://ChestCTScanClassifier/Staging")

# Load production model for serving
production_model = mlflow.keras.load_model("models://ChestCTScanClassifier/Production")
```

## Model Lifecycle:
1. **None** → Model just registered
2. **Staging** → Model ready for testing (accuracy ≥ 90%)
3. **Production** → Model approved for production use (accuracy ≥ 95%)
4. **Archived** → Previous versions moved to archive

## Current Model Status:
- Name: ChestCTScanClassifier
- Version: Not registered
- Test Accuracy: 97.01%
- Recommended Stage: Production
