import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.tensorflow
from urllib.parse import urlparse
import dagshub
from mlflow_e2e.entity.config_entity import EvaluationConfig
from mlflow_e2e.utils.common import save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        """
        Generate validation data for model evaluation
        """
        datagenerator_kwargs = dict(rescale=1.0 / 255, validation_split=0.30)

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

    def evaluation(self):
        """
        Evaluate the model and return scores
        """
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """
        Save evaluation scores to JSON file
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def load_model(self, path: Path) -> tf.keras.Model:
        """
        Load the trained model
        """
        return tf.keras.models.load_model(path)

    def log_into_mlflow(self):
        """
        Log model metrics and artifacts into MLflow
        """
        # Initialize DagHub integration
        dagshub.init(repo_owner="yahiaehab10", repo_name="mlflow-e2e", mlflow=True)

        # Set MLflow tracking URI
        mlflow.set_registry_uri(self.config.mlflow_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config.all_params)

            # Log metrics
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # Log model artifacts (saved model files) instead of using MLflow model logging
            # This avoids the keras version compatibility issue
            mlflow.log_artifact(str(self.config.path_of_model), "model_artifacts")
            mlflow.log_artifact("scores.json", "evaluation_metrics")

            # Log model summary information
            mlflow.log_text(
                f"Model trained on chest CT scan data with {self.config.params_batch_size} batch size",
                "model_info.txt",
            )
