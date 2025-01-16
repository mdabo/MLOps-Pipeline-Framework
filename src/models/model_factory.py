# => src/models/model_factory.py

from typing import Dict, Any
from enum import Enum, auto
from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class LearningType(Enum):
    SUPERVISED_CLASSIFICATION = auto()
    SUPERVISED_REGRESSION = auto()
    UNSUPERVISED_CLUSTERING = auto()
    UNSUPERVISED_DIMENSION_REDUCTION = auto()


class ModelFactory:
    """
    Factory class for creating different types of models.
    """

    @staticmethod
    def create_model(learning_type: LearningType, config: Dict[str, Any]) -> BaseModel:
        if learning_type == LearningType.SUPERVISED_CLASSIFICATION:
            return SupervisedClassifier(config)
        elif learning_type == LearningType.SUPERVISED_REGRESSION:
            return SupervisedRegressor(config)
        elif learning_type == LearningType.UNSUPERVISED_CLUSTERING:
            return UnsupervisedClustering(config)
        elif learning_type == LearningType.UNSUPERVISED_DIMENSION_REDUCTION:
            return UnsupervisedDimensionReduction(config)
        else:
            raise ValueError(f"Unsupported learning type: {learning_type}")


class SupervisedClassifier(BaseModel):
    """Implementation for supervised classification."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = "classifier"
        self.learning_type = "supervised"

        model_config = config.get("training", {}).get("model", {})
        model_type = model_config.get("type", "random_forest")
        model_params = model_config.get("params", {})

        if model_type == "random_forest":
            print("\nMODEL FACTORY: Random Forest Classifier =>", model_params)
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        train_score = self.model.score(X, y)
        self.log_training_step({"train_score": train_score})

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)


class SupervisedRegressor(BaseModel):
    """Implementation for supervised regression."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = "regressor"
        self.learning_type = "supervised"

        model_config = config.get("training", {}).get("model", {})
        model_type = model_config.get("type", "random_forest")
        model_params = model_config.get("params", {})

        if model_type == "random_forest":
            print("\nMODEL FACTORY: Random Forest Regressor =>", model_params)
            self.model = RandomForestRegressor(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        train_score = self.model.score(X, y)
        self.log_training_step({"train_score": train_score})

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)


class UnsupervisedClustering(BaseModel):
    """Implementation for unsupervised clustering."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = "clustering"
        self.learning_type = "unsupervised"

        model_config = config.get("training", {}).get("model", {})
        model_type = model_config.get("type", "kmeans")
        model_params = model_config.get("params", {})

        if model_type == "kmeans":
            print("\nMODEL FACTORY: KMeans =>", model_params)
            self.model = KMeans(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y=None):
        self.model.fit(X)
        self.is_fitted = True
        inertia = self.model.inertia_
        self.log_training_step({"inertia": inertia})

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)


class UnsupervisedDimensionReduction(BaseModel):
    """Implementation for dimensionality reduction."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_type = "dimension_reduction"
        self.learning_type = "unsupervised"

        model_config = config.get("training", {}).get("model", {})
        model_type = model_config.get("type", "pca")
        model_params = model_config.get("params", {})

        if model_type == "pca":
            print("\nMODEL FACTORY: PCA =>", model_params)
            self.model = PCA(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y=None):
        self.model.fit(X)
        self.is_fitted = True
        explained_variance = self.model.explained_variance_ratio_.sum()
        self.log_training_step({"explained_variance": explained_variance})

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        return self.model.transform(X)
