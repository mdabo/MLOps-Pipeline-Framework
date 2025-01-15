
import dataclasses
from typing import Dict, List, Tuple
from typing import Optional

class FeatureMetadata:
    """Metadata for a dataset feature."""
    name: str
    dtype: str
    missing_count: int
    unique_count: int
    numeric_stats: Optional[Dict] = None
    categorical_stats: Optional[Dict] = None
    
    def __init__(self, name: str, dtype: str, missing_count: int, unique_count: int, numeric_stats: Optional[Dict] = None, categorical_stats: Optional[Dict] = None):
        """
        Initialize the metadata object.
        
        Args:
            name: Name of the feature
            dtype: Data type of the feature
            missing_count: Number of missing values
            unique_count: Number of unique values
            numeric_stats: Dictionary of numeric feature statistics
            categorical_stats: Dictionary of categorical feature statistics
        """
        self.name = name
        self.dtype = dtype
        self.missing_count = missing_count
        self.unique_count = unique_count
        self.numeric_stats = numeric_stats
        self.categorical_stats = categorical_stats