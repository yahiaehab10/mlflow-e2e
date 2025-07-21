# 🚀 Quick Start Guide - MLflow Chest CT Classification

## Prerequisites

- Python 3.11
- Conda environment: `e2e`

## ⚡ Quick Commands

### 1. Setup

```bash
conda activate e2e
pip install -r requirements.txt
pip install -e .
```

### 2. Run Complete Pipeline

```bash
conda activate e2e
python main.py
```

### 3. View MLflow Experiments

```bash
python mlflow_demo.py
```

### 4. Project Summary

```bash
python project_summary.py
```

## 📊 What Each Script Does

| Script               | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `main.py`            | Runs the complete ML pipeline (4 stages) |
| `mlflow_demo.py`     | Views MLflow experiments and statistics  |
| `project_summary.py` | Shows comprehensive project overview     |

## 🔧 Configuration Files

| File                 | Purpose                       |
| -------------------- | ----------------------------- |
| `config/config.yaml` | Paths and directory structure |
| `params.yaml`        | Model hyperparameters         |
| `requirements.txt`   | Python dependencies           |

## 📈 Experiment Tracking

All experiments are automatically tracked with:

- ✅ Parameters (batch size, epochs, learning rate, etc.)
- ✅ Metrics (accuracy, loss)
- ✅ Artifacts (model files, evaluation scores)
- ✅ Remote storage on DagHub

## 🎯 Model Performance

Current results (1 epoch demo):

- **Accuracy**: ~43%
- **Loss**: ~5.7

To improve performance:

1. Increase `EPOCHS` in `params.yaml`
2. Adjust learning rate
3. Fine-tune other hyperparameters

## 🔗 Links

- **DagHub MLflow**: https://dagshub.com/yahiaehab10/mlflow-e2e
- **Local MLflow**: Run `mlflow ui` after training

## 🏥 Dataset

- **Source**: Chest CT Scan images
- **Classes**: Normal vs Adenocarcinoma
- **Size**: ~49MB (auto-downloaded)
- **Architecture**: VGG16 Transfer Learning

---

**🎉 Happy Experimenting!** 🧪🔬
