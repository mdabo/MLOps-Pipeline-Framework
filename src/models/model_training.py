import os
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
import logging
from datetime import datetime
from joblib import dump, load

from .model_factory import ModelFactory, LearningType

class ModelTrainer:
    """
    Handles model training + evaluation for supervised & unsupervised tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.training_history = []
        self.learning_type = None

        # If you want to keep a reference to the same vectorizer here:
        self.vectorizer = None
        self.vectorizer_path = None

    def _determine_learning_type(self, y: Optional[pd.Series]) -> LearningType:
        """
        Decide classification vs regression vs unsupervised based on 'y'.
        """
        if y is None:
            task = self.config.get("unsupervised_learning_task", "clustering")
            if task == "clustering":
                print("\nMODEL TRAINER: UNSUPERVISED_CLUSTERING used")
                return LearningType.UNSUPERVISED_CLUSTERING
            else:
                print("\nMODEL TRAINER: UNSUPERVISED_DIMENSION_REDUCTION used")
                return LearningType.UNSUPERVISED_DIMENSION_REDUCTION
        else:
            # classification vs regression
            if y.dtype in ["int64", "bool"] or y.nunique() < 10:
                print("\nMODEL TRAINER: SUPERVISED_CLASSIFICATION used")
                return LearningType.SUPERVISED_CLASSIFICATION
            else:
                print("\nMODEL TRAINER: SUPERVISED_REGRESSION used")
                return LearningType.SUPERVISED_REGRESSION

    def prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: float = 0.2,
    ) -> Tuple:
        """
        Splits into train/test. If no y => unsupervised => only X_train, X_test returned.
        """
        random_state = self.config.get("training", {}).get("model", {}).get("params", {}).get("random_state", 42)
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
            return X_train, X_test

    def train_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                    learning_type: Optional[LearningType] = None) -> Dict[str, float]:
        """
        Train the model with a train/test split and evaluate. 
        Return the performance metrics.
        """
        try:
            # 1) Determine learning type
            self.learning_type = learning_type or self._determine_learning_type(y)
            self.logger.info(f"Training model with type: {self.learning_type}")

            # 2) Create model
            self.model = ModelFactory.create_model(self.learning_type, self.config)

            # 3) Split data
            if y is not None:
                X_train, X_test, y_train, y_test = self.prepare_data(X, y)
                # Fit supervised
                self.model.fit(X_train, y_train)
                metrics = self._evaluate_supervised_model(X_test, y_test)
            else:
                # Unsupervised
                X_train, X_test = self.prepare_data(X)
                self.model.fit(X_train)
                metrics = self._evaluate_unsupervised_model(X_test)

            # 4) Log + store
            self._log_training_results(metrics)
            return metrics

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def _evaluate_supervised_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        y_pred = self.model.predict(X_test)

        if self.learning_type == LearningType.SUPERVISED_CLASSIFICATION:
            return {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
                "f1": f1_score(y_test, y_pred, average="weighted"),
            }
        else:
            return {
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            }

    def _evaluate_unsupervised_model(self, X_test: pd.DataFrame) -> Dict[str, float]:
        if self.learning_type == LearningType.UNSUPERVISED_CLUSTERING:
            predictions = self.model.predict(X_test)
            return {
                "inertia": self.model.model.inertia_,
                "n_clusters": len(np.unique(predictions)),
            }
        else:
            # PCA or other dimension reduction
            transformed = self.model.predict(X_test)
            return {
                "n_components": transformed.shape[1],
                "explained_variance_ratio": np.sum(self.model.model.explained_variance_ratio_),
            }

    def _log_training_results(self, metrics: Dict[str, float]) -> None:
        result = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "model_config": self.config
        }
        self.training_history.append(result)
        self.logger.info(f"Training results => {metrics}")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict[str, float]:
        """
        Run cross-validation. Only for supervised tasks.
        """
        if self.model is None:
            raise ValueError("Model not created/trained yet.")
        if self.learning_type not in [LearningType.SUPERVISED_CLASSIFICATION, LearningType.SUPERVISED_REGRESSION]:
            raise ValueError("Cross-validation is only for supervised tasks.")
        scoring = "accuracy" if self.learning_type == LearningType.SUPERVISED_CLASSIFICATION else "r2"
        scores = cross_val_score(self.model.model, X, y, cv=n_splits, scoring=scoring)
        return {
            "mean_score": scores.mean(),
            "std_score": scores.std(),
            "n_splits": n_splits,
        }

    def save_model(self, save_directory: str, filepath: str) -> None:
        """
        Save trained model + training history. 
        Optionally, also save vectorizer if you store it here.
        """
        if self.model is None:
            raise ValueError("No trained model to save")

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        model_save_path = f"{save_directory}/{filepath}"
        self.model.save_model(model_save_path)

        # Save training history
        history_path = f"{model_save_path}_history.json"
        pd.DataFrame(self.training_history).to_json(history_path)
        self.logger.info(f"Model + history saved => {model_save_path}")

        # If we have a vectorizer reference
        if self.vectorizer is not None and self.vectorizer_path is not None:
            vec_path = f"{save_directory}/{self.vectorizer_path}"
            dump(self.vectorizer, vec_path)
            print(f"Vectorizer saved => {vec_path}")

    def load_model(self, filepath: str) -> None:
        """
        Load a previously saved model + training history.
        Optionally load vectorizer if you track it here.
        """
        if self.model is None:
            # Normally you'd re-create your model from config or something similar
            # Or just do a dummy model so we can call `load_model()`
            raise ValueError("Initialize the model type before loading.")

        self.model.load_model(filepath)
        history_path = f"{filepath}_history.json"
        self.training_history = pd.read_json(history_path).to_dict("records")
        self.logger.info(f"Model + history loaded => {filepath}")

        if self.vectorizer_path and os.path.exists(self.vectorizer_path):
            self.vectorizer = load(self.vectorizer_path)
            print(f"Vectorizer loaded => {self.vectorizer_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.get_feature_importance()

    def get_training_history(self) -> List[Dict[str, Any]]:
        return self.training_history
