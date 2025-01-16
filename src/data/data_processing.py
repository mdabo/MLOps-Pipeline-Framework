# => src/data/data_processing.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import warnings
import time
import yaml

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib  # For saving/loading the TfidfVectorizer
from rich import print, traceback  # for nice console output

from src.features.feature_engineering import FeatureMetadata

traceback.install() # help to see the error in the terminal more clearly


class DataProcessor:
    """Class for data preprocessing, including text vectorization logic."""

    def __init__(self, config_path: str):
        """
        Initialize DataProcessor with config + placeholders for scalers, encoders, vectorizer.
        """
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)

        # Scalers, encoders
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

        self.df: Optional[pd.DataFrame] = None

        # TF-IDF vectorizer (and vocabulary array)
        text_features_config = self.config.get("features", {}).get("text_features", {})
        self.vectorizer = TfidfVectorizer(
            max_features=text_features_config.get("max_features", 100),
            min_df=text_features_config.get("min_df", 0.01),
            max_df=text_features_config.get("max_df", 0.95),
            ngram_range=text_features_config.get("ngram_range", (1, 2)),
        )
        self.vectorizer_vocabulary_ = None

    def _load_config(self, config_path: str) -> Dict[str, any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def data_load(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV or XLSX, auto-infer data types based on first 3000 rows.
        """
        load_params = {
            "low_memory": False,  # reduce dtype warnings
            "on_bad_lines": "warn",
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # partial data for dtype inference
            df_initial = pd.read_csv(file_path, nrows=3000, **load_params)

        print("\nDATA LOADING: Inferring data types based on first 3000 rows...\n")
        dtypes = self.infer_data_types(df_initial)

        #infer data types based on the first 3000 rows. Because sometimes we can have mixed types in the same column
        print("\nDATA LOADING: Data types inferred count\n", df_initial.dtypes.value_counts())

        # Full file read with inferred dtypes
        try:
            if file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path, dtype=dtypes)
            elif file_path.endswith(".xlsx"):
                self.df = pd.read_excel(file_path, dtype=dtypes)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            raise

        return self.df

    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Attempt to infer a better dtype for each column: numeric, datetime, or string.
        """
        dtype_map = {}
        for column in df.columns:
            sample = df[column].dropna().head(1000)
            if len(sample) == 0:
                dtype_map[column] = "object"
                continue

            # If numeric
            if pd.api.types.is_numeric_dtype(sample):
                if pd.api.types.is_integer_dtype(sample):
                    dtype_map[column] = "Int64"
                else:
                    dtype_map[column] = "float64"
            elif pd.api.types.is_datetime64_any_dtype(sample):
                dtype_map[column] = "datetime64[ns]"
            else:
                # Attempt to parse as datetime
                try:
                    pd.to_datetime(sample, format="%Y-%m-%d %H:%M:%S", errors="raise")
                    dtype_map[column] = "datetime64[ns]"
                except:
                    dtype_map[column] = "string"

        return dtype_map

    def process_data(
        self,
        df: pd.DataFrame,
        data_file_name: str = None,
        is_prediction: bool = False,
    ) -> pd.DataFrame:
        """
        Main processing pipeline: handle missing, drop ID, vectorize text, scale numeric, weight, encode cat.
        is_prediction => no re-fitting of transforms (just transform).
        """
        if df is None:
            raise ValueError("No data to process")

        # 1) Missing values
        df_preprocessed = self.handle_missing_values(df)
        print("\nDATA PROCESS: After handling missing values:\n", df_preprocessed)

        # 2) Drop ID columns if configured (only in training)
        if not is_prediction:
            df_preprocessed = self.drop_columns_ids_if_exists(df_preprocessed)

        # 3) Text vectorization
        text_features_config = self.config.get("features", {}).get("text_features", {})
        if text_features_config:
            keep_original = text_features_config.get("keep_original_after_vectorize", False)
            text_columns = text_features_config.get("columns", ["tweet"])
            df_preprocessed = self.vectorize_text_columns(
                df_preprocessed,
                text_columns,
                keep_original=keep_original,
                is_training=(not is_prediction),
            )

        # 4) Scale numeric columns
        df_preprocessed = self.scale_numeric_columns(df_preprocessed)

        # 5) Weighted main features (only training)
        if not is_prediction:
            df_preprocessed = self.give_more_weight_to_main_features(df_preprocessed)

        # 6) Encode categorical
        df_preprocessed = self.generic_encode_categorical_columns(df_preprocessed)

        # 7) Optionally save
        if not is_prediction and data_file_name:
            self.save_processed_data(df_preprocessed, data_file_name)

        return df_preprocessed

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing numeric with mean or median, fill categorical with mode or constant.
        """
        missing_config = self.config["data"]["missing_values"]

        # Numeric
        strategy_numeric = missing_config.get("strategy_numeric", "mean")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print("\nDATA PROCESS: Missing values in numeric cols:\n", df[numeric_cols].isnull().sum())
        if strategy_numeric == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy_numeric == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(
            "\nDATA PROCESS: Missing numeric after fill:\n", df[numeric_cols].isnull().sum()
        )

        # Categorical
        strategy_categorical = missing_config.get("strategy_categorical", "mode")
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        print("\nDATA PROCESS: Categorical cols:\n", cat_cols)
        if strategy_categorical == "mode":
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        elif strategy_categorical == "constant":
            df[cat_cols] = df[cat_cols].fillna("")

        return df

    def drop_columns_ids_if_exists(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop the ID column by index (from config) if it is valid.
        """
        col_index = self.config["data"]["row_data_column_id_index"]
        print("\nDATA PROCESS: row_data_column_id_index:\n", col_index)

        if col_index != -1 and col_index < len(df.columns):
            print("\nDATA PROCESS: Dropping ID column => ", df.columns[col_index])
            df = df.drop(df.columns[col_index], axis=1)
        else:
            print("\nDATA PROCESS: No valid ID col to drop or none specified.")

        return df

    def vectorize_text_columns(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        keep_original: bool = False,
        is_training: bool = True,
    ) -> pd.DataFrame:
        """
        Vectorize text. If is_training, fit_transform. Else transform (use existing vocabulary).
        """
        if not text_columns:
            print("DATA PROCESS: No text columns to vectorize.")
            return df

        # Download NLTK data if needed
        try:
            nltk.download("punkt")
            nltk.download("stopwords")
            nltk.download("wordnet")
            nltk.download("omw-1.4")
        except Exception as e:
            self.logger.warning(f"NLTK resource error: {e}")

        result_df = df.copy()

        for col in text_columns:
            if col not in result_df.columns:
                print(f"DATA PROCESS: Text column '{col}' not found, skipping.")
                continue

            print(f"\nDATA PROCESS (vectorization): Processing column '{col}'...")

            processed_col_name = f"{col}_processed"
            result_df[processed_col_name] = result_df[col].apply(self._clean_text)

            # Fit or transform
            if is_training:
                sparse_matrix = self.vectorizer.fit_transform(result_df[processed_col_name])
                self.vectorizer_vocabulary_ = self.vectorizer.get_feature_names_out()
            else:
                sparse_matrix = self.vectorizer.transform(result_df[processed_col_name])

            feature_names = self.vectorizer.get_feature_names_out()
            # Build new DataFrame
            vectorized_df = pd.DataFrame(
                sparse_matrix.toarray(),
                columns=[f"{col}_feature_{word}" for word in feature_names],
                index=result_df.index,
            )
            # Merge
            result_df = pd.concat([result_df, vectorized_df], axis=1)

            # Clean up
            result_df.drop(columns=[processed_col_name], inplace=True)
            if not keep_original:
                result_df.drop(columns=[col], inplace=True)

            print(f"DATA PROCESS (vectorization): Created {len(feature_names)} features.")

        return result_df

    def _clean_text(self, text) -> str:
        """
        Lowercase, tokenize, remove stopwords, lemmatize, and remove short tokens.
        """
        if pd.isna(text):
            return ""
        try:
            text = str(text).lower()
            tokens = word_tokenize(text)

            lemmatizer = WordNetLemmatizer()
            stops = set(stopwords.words("english"))

            cleaned_tokens = []
            for token in tokens:
                if token.isalnum() and token not in stops and len(token) > 1:
                    cleaned_tokens.append(lemmatizer.lemmatize(token))

            return " ".join(cleaned_tokens)
        except Exception as e:
            print(f"\nDATA PROCESS: Error cleaning text => {e}")
            return ""

    def scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standard-scaler on numeric columns, fitted only once if training mode.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print("\nDATA PROCESS: Numeric cols to scale => ", numeric_cols)

        for col in numeric_cols:
            if col not in self.scalers:
                scaler = StandardScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                df[[col]] = self.scalers[col].transform(df[[col]])

        return df

    def generic_encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encode all remaining categorical columns, fitted only once in training mode.
        """
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(cat_cols) == 0:
            return df

        print("\nDATA PROCESS: Categorical columns to encode =>", list(cat_cols))

        keep_original = self.config["data"].get("keep_original_column_data_after_encode", False)

        for col in cat_cols:
            if col not in self.encoders:
                le = LabelEncoder()
                df[f"{col}_categorical_encoded"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                df[f"{col}_categorical_encoded"] = self.encoders[col].transform(df[col].astype(str))

        if not keep_original:
            df.drop(columns=cat_cols, inplace=True)

        return df

    def give_more_weight_to_main_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multiply certain features by a weighting factor from config.
        """
        main_feats = self.config["features"].get("main_features", None)
        if not main_feats:
            print("\nDATA PROCESS: No main features to weight, skipping.")
            return df

        weight = self.config["features"].get("main_features_weight", 1.0)
        print("\nDATA PROCESS: Weighting features =>", main_feats, "by", weight)
        for feat in main_feats:
            if feat in df.columns:
                df[feat] = df[feat] * weight
                print(f"DATA PROCESS: Weighted feature '{feat}' updated.")

        return df

    def save_processed_data(self, df: pd.DataFrame, data_file_name: str) -> None:
        """
        Save the processed data to CSV or XLSX in processed_data_path from config.
        """
        out_path = self.config["data"]["processed_data_path"]
        if data_file_name.endswith(".csv"):
            df.to_csv(f"{out_path}processed_{data_file_name}", index=False)
        elif data_file_name.endswith(".xlsx"):
            df.to_excel(f"{out_path}processed_{data_file_name}", index=False)
        else:
            df.to_csv(f"{out_path}processed_{data_file_name}.csv", index=False)

        print("\nDATA PROCESS: Processed data saved successfully!")

    def save_vectorizer(self, filepath: str) -> None:
        """
        Save the current TfidfVectorizer to disk (with vocabulary).
        """
        joblib.dump(self.vectorizer, filepath)
        print(f"\nDATA PROCESS: TF-IDF vectorizer saved to {filepath}")

    def load_vectorizer(self, filepath: str) -> None:
        """
        Load a previously fitted TfidfVectorizer from disk.
        """
        self.vectorizer = joblib.load(filepath)
        # Store the vocabulary array for reference
        self.vectorizer_vocabulary_ = self.vectorizer.get_feature_names_out()
        print(f"\nDATA PROCESS: TF-IDF vectorizer loaded from {filepath}")

    def get_data_frame(self) -> pd.DataFrame:
        """
        Return the internally loaded DataFrame.
        """
        return self.df


# Classes below remain largely unchanged, but are included for completeness.

class FeatureStats:
    """Stores stats for each feature, used by DataAnalyzer class."""
    def __init__(
        self,
        name: str,
        dtype: str,
        missing_values_count: int,
        unique_values_count: int,
        numeric_stats: Optional[Dict] = None,
        categorical_stats: Optional[Dict] = None,
    ):
        self.name = name
        self.dtype = dtype
        self.missing_values_count = missing_values_count
        self.unique_values_count = unique_values_count
        self.numeric_stats = numeric_stats
        self.categorical_stats = categorical_stats


class DataAnalyzer:
    """Class for automated exploratory data analysis."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.features: List[FeatureMetadata] = []
        self.feature_stats: Dict[str, FeatureStats] = {}
        self._analyze_features()

    def _analyze_features(self) -> None:
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            unique_count = self.df[column].nunique()

            if pd.api.types.is_numeric_dtype(self.df[column]):
                numeric_stats = {
                    "mean": self.df[column].mean(),
                    "std": self.df[column].std(),
                    "min": self.df[column].min(),
                    "max": self.df[column].max(),
                    "skew": self.df[column].skew(),
                    "kurtosis": self.df[column].kurtosis(),
                    "median": self.df[column].median(),
                }
                self.feature_stats[column] = FeatureStats(
                    column,
                    "numeric",
                    missing_count,
                    unique_count,
                    numeric_stats=numeric_stats,
                )
            else:
                # Already encoded => might not have many "categorical" left,
                # but let's keep logic here for general usage:
                val_counts = self.df[column].value_counts()
                categorical_stats = {
                    "mode": self.df[column].mode()[0],
                    "entropy": self._calculate_entropy(val_counts),
                }
                self.feature_stats[column] = FeatureStats(
                    column,
                    "categorical",
                    missing_count,
                    unique_count,
                    categorical_stats=categorical_stats,
                )

        print("\nDATA ANALYSE: number of features analyzed =>", len(self.feature_stats))

    def generate_summary(self) -> Dict:
        print("\nDATA ANALYSE: generating summary...\n")
        return {
            "dataset_shape": self.df.shape,
            "total_missing": self.df.isnull().sum().sum(),
            "numeric_features": len(
                [f for f in self.feature_stats.values() if f.dtype == "numeric"]
            ),
            "categorical_features": len(
                [f for f in self.feature_stats.values() if f.dtype == "categorical"]
            ),
            "feature_stats": {
                name: vars(stats) for name, stats in self.feature_stats.items()
            },
        }

    def plot_correlations(self, save_path: Optional[str] = None) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        features_names = self.df.columns
        if len(features_names) > 15:
            print(
                "\nDATA ANALYSE: Too many features to plot correlation. Limiting to 10 randomly."
            )
            features_names = np.random.choice(features_names, 10, replace=False)

        print("\nDATA ANALYSE: features_names to plot =>\n", features_names)
        corr_matrix = self.df[features_names].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("\n DATA ANALYSE: Feature Correlations")
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _calculate_entropy(value_counts: pd.Series) -> float:
        proportions = value_counts / value_counts.sum()
        return -np.sum(proportions * np.log2(proportions))

    def plot_distributions(self, max_features: int = 5) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric_features = [
            name for name, stats in self.feature_stats.items() if stats.dtype == "numeric"
        ][:max_features]

        fig, axes = plt.subplots(len(numeric_features), 2, figsize=(12, 4 * len(numeric_features)))
        for i, feature in enumerate(numeric_features):
            sns.histplot(data=self.df, x=feature, ax=axes[i, 0])
            axes[i, 0].set_title(f"{feature} Distribution")

            sns.boxplot(data=self.df, x=feature, ax=axes[i, 1])
            axes[i, 1].set_title(f"{feature} Box Plot")

        plt.tight_layout()
        plt.show()

    def analyze_pca_components(self, n_components: int = 2):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        numeric_features = [
            name for name, stats in self.feature_stats.items() if stats.dtype == "numeric"
        ]
        if len(numeric_features) < 2:
            raise ValueError("Need at least 2 numeric features for PCA")

        X = self.df[numeric_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        explained_variance = pca.explained_variance_ratio_

        return X_pca, explained_variance, pca.components_
