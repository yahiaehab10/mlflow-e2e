"""
MLflow Experiment Viewer

View your MLflow experiments and model performance
"""

import mlflow
import dagshub
import json
import os


def main():
    print("🔬 MLflow Chest CT Classification - Experiment Viewer")
    print("=" * 55)

    # Initialize DagHub
    dagshub.init(repo_owner="yahiaehab10", repo_name="mlflow-e2e", mlflow=True)

    try:
        # Get recent experiments
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=3)

        if len(runs) > 0:
            print(f"📊 Latest {len(runs)} Experiments:")
            print("-" * 40)

            for idx, run in runs.iterrows():
                print(f"🔬 Experiment {idx + 1}:")
                print(f"   Status: {run['status']}")
                if "metrics.accuracy" in run:
                    print(f"   Accuracy: {run['metrics.accuracy']:.4f}")
                if "metrics.loss" in run:
                    print(f"   Loss: {run['metrics.loss']:.4f}")
                print()

        # Show current results
        if os.path.exists("scores.json"):
            with open("scores.json", "r") as f:
                scores = json.load(f)

            print("🎯 Current Model Performance:")
            print(
                f"   Accuracy: {scores['accuracy']:.4f} ({scores['accuracy']*100:.2f}%)"
            )
            print(f"   Loss: {scores['loss']:.4f}")

        print("\n💡 View all experiments:")
        print("   🌐 DagHub: https://dagshub.com/yahiaehab10/mlflow-e2e")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Run the training pipeline first: python main.py")


if __name__ == "__main__":
    main()
