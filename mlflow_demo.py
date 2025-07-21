"""
MLflow Experiment Tracking Demo

This script demonstrates how to view and analyze the MLflow experiments
that were logged during the training and evaluation process.
"""

import mlflow
import mlflow.tensorflow
import dagshub

# Initialize DagHub integration
dagshub.init(repo_owner="yahiaehab10", repo_name="mlflow-e2e", mlflow=True)


def view_latest_experiments():
    """
    View the latest MLflow experiments
    """
    print("🔍 Viewing Latest MLflow Experiments")
    print("=" * 50)

    # Get the latest runs
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=5)

    if len(runs) > 0:
        print(f"Found {len(runs)} recent experiments:")
        print()

        for idx, run in runs.iterrows():
            print(f"📊 Experiment {idx + 1}:")
            print(f"   Run ID: {run['run_id']}")
            print(f"   Status: {run['status']}")
            print(f"   Start Time: {run['start_time']}")

            # Display metrics
            if "metrics.accuracy" in run and run["metrics.accuracy"] is not None:
                print(f"   Accuracy: {run['metrics.accuracy']:.4f}")
            if "metrics.loss" in run and run["metrics.loss"] is not None:
                print(f"   Loss: {run['metrics.loss']:.4f}")

            # Display parameters
            params = [col for col in run.index if col.startswith("params.")]
            if params:
                print("   Parameters:")
                for param in params[:5]:  # Show first 5 parameters
                    param_name = param.replace("params.", "")
                    print(f"     {param_name}: {run[param]}")

            print("-" * 30)
    else:
        print("No experiments found. Run the training pipeline first!")


def get_experiment_summary():
    """
    Get a summary of all experiments
    """
    print("\n📈 Experiment Summary")
    print("=" * 50)

    runs = mlflow.search_runs()

    if len(runs) > 0:
        print(f"Total Experiments: {len(runs)}")

        # Get accuracy statistics
        if "metrics.accuracy" in runs.columns:
            accuracy_stats = runs["metrics.accuracy"].describe()
            print(f"\nAccuracy Statistics:")
            print(f"  Mean: {accuracy_stats['mean']:.4f}")
            print(f"  Best: {accuracy_stats['max']:.4f}")
            print(f"  Worst: {accuracy_stats['min']:.4f}")

        # Get loss statistics
        if "metrics.loss" in runs.columns:
            loss_stats = runs["metrics.loss"].describe()
            print(f"\nLoss Statistics:")
            print(f"  Mean: {loss_stats['mean']:.4f}")
            print(f"  Best: {loss_stats['min']:.4f}")
            print(f"  Worst: {loss_stats['max']:.4f}")
    else:
        print("No experiments found!")


if __name__ == "__main__":
    print("🚀 MLflow Chest CT Scan Classification - Experiment Viewer")
    print("=" * 60)

    try:
        view_latest_experiments()
        get_experiment_summary()

        print("\n💡 Tips:")
        print(
            "- View experiments on DagHub: https://dagshub.com/yahiaehab10/mlflow-e2e"
        )
        print("- Run 'mlflow ui' for local MLflow interface")
        print("- Check scores.json for latest evaluation results")

    except Exception as e:
        print(f"❌ Error accessing MLflow: {e}")
        print("Make sure you've run the training pipeline first!")
