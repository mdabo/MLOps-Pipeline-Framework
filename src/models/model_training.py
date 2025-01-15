import os
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, mean_squared_error, r2_score)
import logging
from datetime import datetime
from .model_factory import ModelFactory, LearningType

class ModelTrainer:
    """
    Class to handle model training and evaluation for both supervised
    and unsupervised learning scenarios.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.training_history = []
        self.learning_type = None
    
    def _determine_learning_type(self, y: Optional[pd.Series]) -> LearningType:
        """
        Determine the type of learning task based on data and config.
        
        Args:
            y: Target variable (if any)
            
        Returns:
            LearningType enum value
        """
        if y is None:
            # Unsupervised learning
            task = self.config.get('unsupervised_learning_task', 'clustering')
            if task == 'clustering':
                print('\n MODEL TRAINER: UNSUPERVISED_CLUSTERING algorithm will be used')
                return LearningType.UNSUPERVISED_CLUSTERING
            else:
                print('\n MODEL TRIANER: UNSUPERVISED_DIMENSION_REDUCTION algorithm will be used')
                return LearningType.UNSUPERVISED_DIMENSION_REDUCTION
        else:
            # Supervised learning
            if y.dtype in ['int64', 'bool'] or y.nunique() < 10:
                print('\n MODEL TRAINER: SUPERVISED_CLASSIFICATION algorithm will be used')
                return LearningType.SUPERVISED_CLASSIFICATION
            else:
                print('\n MODEL TRAINER: SUPERVISED_REGRESSION algorithm will be used')
                return LearningType.SUPERVISED_REGRESSION
    
    def prepare_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: float = 0.2
    ) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of train/test splits
        """
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.config.get('training').get('model').get('params').get('random_state')
            )
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(
                X,
                test_size=test_size,
                random_state=self.config.get('training').get('model').get('params').get('random_state')
            )
            return X_train, X_test
    
    def _evaluate_supervised_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate supervised model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        
        if self.learning_type == LearningType.SUPERVISED_CLASSIFICATION:
            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:  # Regression
            return {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
    
    def _evaluate_unsupervised_model(self, X_test: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate unsupervised model performance.
        
        Args:
            X_test: Test features
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.learning_type == LearningType.UNSUPERVISED_CLUSTERING:
            predictions = self.model.predict(X_test)
            return {
                'inertia': self.model.model.inertia_,
                'n_clusters': len(np.unique(predictions))
            }
        else:  # Dimension reduction
            transformed = self.model.predict(X_test)
            return {
                'n_components': transformed.shape[1],
                'explained_variance_ratio': np.sum(self.model.model.explained_variance_ratio_)
            }
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        learning_type: Optional[LearningType] = None
    ) -> Dict[str, float]:
        """
        Train model based on data and learning type.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            learning_type: Type of learning task (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Determine learning type if not specified
            self.learning_type = learning_type or self._determine_learning_type(y)
            
            self.logger.info(f"Training model with learning type: {self.learning_type}")
            
            # Create model
            self.model = ModelFactory.create_model(self.learning_type, self.config)
            
            # Prepare data
            if y is not None:
                X_train, X_test, y_train, y_test = self.prepare_data(X, y)
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Evaluate
                metrics = self._evaluate_supervised_model(X_test, y_test)
            else:
                X_train, X_test = self.prepare_data(X)
                
                # Train model
                self.model.fit(X_train)
                
                # Evaluate
                metrics = self._evaluate_unsupervised_model(X_test)
            
            # Log results
            self._log_training_results(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise
    
    def _log_training_results(self, metrics: Dict[str, float]) -> None:
        """Log training results with timestamp."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_config': self.config
        }
        self.training_history.append(result)
        
        self.logger.info(f"Training results: {metrics}")
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation for supervised models.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV splits
            
        Returns:
            Dictionary of cross-validation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train_model first.")
            
        if self.learning_type not in [LearningType.SUPERVISED_CLASSIFICATION,
                                    LearningType.SUPERVISED_REGRESSION]:
            raise ValueError("Cross-validation only supported for supervised learning")
        
        scores = cross_val_score(
            self.model.model,
            X, y,
            cv=n_splits,
            scoring='accuracy' if self.learning_type == LearningType.SUPERVISED_CLASSIFICATION
            else 'r2'
        )
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n_splits': n_splits
        }
    
    def save_model(self, save_directory, filepath: str) -> None:
        """Save trained model and training history."""
        if self.model is None:
            raise ValueError("No trained model to save")

        # if save_directory don't exist, create it
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save model
        self.model.save_model(f"{save_directory}/{filepath}")
        
        # Save training history separately
        history_path = f"{save_directory}/{filepath}_history.json"
        pd.DataFrame(self.training_history).to_json(history_path)
        
        self.logger.info(f"Model and history saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model and training history."""
        if self.model is None:
            raise ValueError("Initialize model type before loading")
            
        self.model.load_model(filepath)
        
        # Load training history
        history_path = f"{filepath}_history.json"
        self.training_history = pd.read_json(history_path).to_dict('records')
        
        self.logger.info(f"Model and history loaded from {filepath}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.get_feature_importance()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get complete training history."""
        return self.training_history