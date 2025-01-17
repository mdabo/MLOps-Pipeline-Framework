# => src/models/base_model.py

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union, List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import logging
from datetime import datetime


class BaseModel(ABC):
    """
    Abstract base class for all models (supervised and unsupervised).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[BaseEstimator] = None
        self.model_type: str = "undefined"
        self.learning_type: str = "undefined"
        self.is_fitted: bool = False
        self.logger = logging.getLogger(self.__class__.__name__)

        self.training_history: Dict[str, List] = {
            "metrics": [],
            "timestamps": [],
            "parameters": []
        }

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]] = None) -> None:
        """Fit the model to the data."""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions using the trained model."""
        pass

    def save_model(self, filepath: str) -> None:
        """Save model & metadata to disk."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{filepath}_{timestamp}"

        model_data = {
            "model": self.model,
            "config": self.config,
            "training_history": self.training_history,
            "model_type": self.model_type,
            "learning_type": self.learning_type,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
        print(f"Model saved to {model_path}")

    def load_model(self, filepath: str) -> None:
        """Load model & metadata from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.config = model_data["config"]
        self.training_history = model_data["training_history"]
        self.model_type = model_data["model_type"]
        self.learning_type = model_data["learning_type"]
        self.is_fitted = model_data["is_fitted"]
        self.logger.info(f"Model loaded from {filepath}")

    def get_params(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not initialized")
        return self.model.get_params()

    def set_params(self, **params) -> None:
        if self.model is None:
            raise ValueError("Model not initialized")
        self.model.set_params(**params)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature_importances_ if supported, else None."""
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}
        return None

    def log_training_step(self, metrics: Dict[str, float]) -> None:
        """Log training progress."""
        self.training_history["metrics"].append(metrics)
        self.training_history["timestamps"].append(datetime.now().isoformat())
        self.training_history["parameters"].append(self.get_params())

    def get_training_history(self) -> Dict[str, List]:
        """Return stored training history."""
        return self.training_history

    def summary(self) -> str:
        """Summary of the model."""
        lines = [
            f"Model Type: {self.model_type}",
            f"Learning Type: {self.learning_type}",
            f"Fitted: {self.is_fitted}",
            "\nConfiguration:",
            str(self.config),
            "\nTraining History:",
            str(self.training_history),
        ]
        return "\n".join(lines)
