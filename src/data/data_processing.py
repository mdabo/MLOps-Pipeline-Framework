import pandas as pd
import numpy as np
import dataclasses
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.features.feature_engineering import FeatureMetadata
import yaml
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import logging
import time
from rich import print, traceback #print will override the built-in print function vith more details and clear output

# Download the 'punkt' tokenizer
import nltk


traceback.install() # help to see the error in the terminal more clearly

class DataProcessor:
    """Class for data preprocessing."""

    def __init__(self, config_path: str) :
        
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        self.scalers: Dict[str, StandardScaler] = {} #To scale all numeric features
        self.encoders: Dict[str, LabelEncoder] = {}
        self.df = None
        
        # Initialize vectorizer with configuration
        text_features_config = self.config.get('features', {}).get('text_features', {})
        self.vectorizer = TfidfVectorizer(
            max_features=text_features_config.get('max_features', 100),
            min_df=text_features_config.get('min_df', 0.01),
            max_df=text_features_config.get('max_df', 0.95),
            ngram_range=text_features_config.get('ngram_range', (1, 2))
        )

        # vectorizer_vocabulary_ a ndarray of str objects
        self.vectorizer_vocabulary_  = None
    
    def _load_config(self, config_path: str) -> Dict[str, any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def data_load(self, file_path: str) -> pd.DataFrame:
        """Load data from a file."""
        load_params = {
            'low_memory': False,  # Prevent mixed type inference warnings
            'on_bad_lines': 'warn',  # Don't fail on problematic lines
        }
        # declare df_initial to infer data types
        df_initial = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df_initial = pd.read_csv(file_path, nrows=3000, **load_params)
            
        print('\nDATA LOADING: Infer data types based on the first 1000 rows \n')
        #infer data types based on the first 1000 rows. Because sometimes we can have mixed types in the same column
        dtypes = self.infer_data_types(df_initial)
        
        #print the data types inferred count
        print('\nDATA LOADING: Data types inferred count\n', df_initial.dtypes.value_counts())
        
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, dtype=dtypes)
            elif file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path, dtype=dtypes)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {e}")
            
            raise
        
        return self.df

    def process_data(self, df: pd.DataFrame, data_file_name: str = None, is_prediction: bool = False):
        """
        Process data with scaling and encoding.
        Args:
            df: DataFrame to process
            data_file_name: the name of the file to be processed (training or test data). Used for saving.
            is_prediction: whether this is for prediction (True) or training/testing (False)
        """
        if df is None:
            raise ValueError("No data to process")
        
        # 1. Handle missing values
        df_preprocessed = self.handle_missing_values(df)
        print('\nDATA PROCESS: Data after handling missing values\n', df_preprocessed)
        
        # 2. Drop ID columns if configured and not in prediction mode
        if not is_prediction:  # Only drop IDs during training/testing
            df_preprocessed = self.drop_columns_ids_if_exists(df_preprocessed)
        
        # 3. Text vectorization
        text_features_config = self.config.get('features', {}).get('text_features', {})
        if text_features_config:
            keep_original = text_features_config.get('keep_original_after_vectorize', False)
            text_columns = text_features_config.get('columns', ["tweet"])
            df_preprocessed = self.vectorize_text_columns(
                df_preprocessed, 
                text_columns,
                keep_original=keep_original
            )
        
        # 4. Scale numeric columns
        df_preprocessed = self.scale_numeric_columns(df_preprocessed)
        
        # 5. Apply main features weight if not in prediction mode
        if not is_prediction:
            df_preprocessed = self.give_more_weight_to_main_features(df_preprocessed)
        
        # 6. Encode categorical columns
        df_preprocessed = self.generic_encode_categorical_columns(df_preprocessed)
        
        # 7. Save processed data only if not in prediction mode
        if not is_prediction and data_file_name:
            self.save_processed_data(df_preprocessed, data_file_name)
        
        return df_preprocessed

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
         
         # Access configuration in DataProcessor class
        missing_config = self.config['data']['missing_values']
        
        strategy_numeric = missing_config.get('strategy_numeric', 'mean')  # 'mean' is the default value if 'strategy_numeric' is not found
        if strategy_numeric == 'mean':
            #get numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            #print('\nnumeric_columns\n', numeric_columns)
            #total missing values in numeric columns
            print('\nDATA PROCESS: missing values in numeric columns\n', df[numeric_columns].isnull().sum())
            df.loc[:, numeric_columns] = df.loc[:, numeric_columns].fillna(df[numeric_columns].mean())
            print('\nDATA PROCESS: Missing values in numeric columns after treatment \n', df[numeric_columns].isnull().sum())
            
        elif strategy_numeric == 'median':
            df.loc[:, numeric_columns] = df.loc[:, numeric_columns].fillna(df[numeric_columns].median())
    
        # Handle missing values in categorical columns
        strategy_categorical = missing_config.get('strategy_categorical', 'mode')
        
        if strategy_categorical == 'mode':
            #get categorical columns
            categorical_columns = df.select_dtypes(exclude=[np.number]).columns
            print('\nDATA PROCESS: categorical_columns\n', categorical_columns)
            df.loc[:, categorical_columns] = df.loc[:, categorical_columns].fillna(df[categorical_columns].mode().iloc[0]) #iloc[0] to get the first mode value
        elif strategy_categorical == 'constant':
            df.loc[:, categorical_columns] = df.loc[:, categorical_columns].fillna('')
    
        return df
    
    def drop_columns_ids_if_exists(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # get the columns to drop from the config file
        column_id_index = self.config['data']['row_data_column_id_index']
        print('\nDATA PROCESS: row_data_column_id_index to drop:\n', column_id_index)
        
        if column_id_index != -1:
            print('\nDATA PROCESS: Column with indexes values exist in the conf file. And will be droped: \n', df.columns[column_id_index])
            df = df.drop(df.columns[column_id_index], axis=1)
        else:
            print('DATA PROCESS: No column with indexes values exist in the conf file. Nothing to drop')
            
        return df
            
    def infer_data_types(self, df: pd.DataFrame) -> Dict[str, str]: #used by the data_load method. In order to adapt the data types of the columns to the data
        """
        Automatically infer appropriate data types for DataFrame columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary mapping column names to their specific data types
        """
        dtype_map = {}
        
        for column in df.columns:
            sample = df[column].dropna().head(1000)
            
            if len(sample) == 0:
                dtype_map[column] = 'object'
                continue
                
            if pd.api.types.is_numeric_dtype(sample):
                # Further differentiate between integer and float
                if pd.api.types.is_integer_dtype(sample):
                    dtype_map[column] = 'Int64'  # Using nullable integer type
                else:
                    dtype_map[column] = 'float64'
            elif pd.api.types.is_datetime64_any_dtype(sample):
                dtype_map[column] = 'datetime64[ns]'
            else:
                # Check if it could be converted to datetime
                try:
                    pd.to_datetime(sample, format='%Y-%m-%d %H:%M:%S')
                    dtype_map[column] = 'datetime64[ns]'
                except (ValueError, TypeError):
                    dtype_map[column] = 'string'
        
        return dtype_map

    def scale_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features using StandardScaler."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        #print number of numeric columns
        print('\nDATA PROCESS: number of numeric columns for scalling\n', len(numeric_columns))
        
        for col in numeric_columns:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                df[[col]] = self.scalers[col].fit_transform(df[[col]]) # This is used only for training data. fit: Learns the mean and standard deviation of the data. Then transform: Applies the scaling formula: (x - mean) / std
            else:
                df[[col]] = self.scalers[col].transform(df[[col]]) # This ensures test data is scaled exactly the same way as training data
               
        return df   
 
    def generic_encode_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame: # encoding all categorical columns in the same way
        """Encode categorical features using LabelEncoder. All categorical columns are encoded."""
        categorocal_columns = df.select_dtypes(exclude=[np.number]).columns
        
        print(f'\nDATA PROCESS: Unique values for categorial feature {categorocal_columns[0]}: \n', df[categorocal_columns[0]].unique())
         
        encoded_columns = []
        encoded_data = []
        for col in categorocal_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()# Can be One-Hot encoder, Ordinal encoder, etc. todo: Define the encoder in the config file
                encoded_col = self.encoders[col].fit_transform(df[col].astype(str))
            else:
                encoded_col = self.encoders[col].transform(df[col].astype(str))
            
            encoded_data.append(encoded_col)
            encoded_columns.append(f'{col}_categorical_encoded')
    
        keep_original_column_data_after_encode = self.config['data']['keep_original_column_data_after_encode']
        if keep_original_column_data_after_encode == False:
            df.drop(categorocal_columns, axis=1, inplace=True)
        
        encoded_data = pd.DataFrame(encoded_data).T
        encoded_data.columns = encoded_columns
        
        print ('\nDATA PROCESS: Categorial encoded_data: \n', encoded_data)
        df = pd.concat([df, encoded_data], axis=1)
        
        return df

    def give_more_weight_to_main_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Apply additional coefficient to the given main features."""
        #get the weight from the config file
        main_features = self.config['features']['main_features']
        
        print('DATA PROCESS: main_features specified to give more weight: ', main_features)
        if main_features is None:
            print('\nDATA PROCESS: No main features (to prioritize by adding coeff) found in config file. Not adding any coeff to any features\n')

        else: 
            main_features_weight = self.config['features']['main_features_weight']
        
            for feature in main_features:
                df[feature] = df[feature] * main_features_weight
                print(f'DATA PROCESS: the feature {feature} is now: \n', df[feature])
        return df
    
    def specific_encode_categorical_columns(
        self, 
        df: pd.DataFrame, 
        column_indices: List[str], 
        encoder_type: str = 'label',
        keep_original: bool = False
    ) -> pd.DataFrame:
        """
        Encode categorical columns using specified encoder.
        
        Args:
            df: Input DataFrame
            column_indices: List of column name to encode
            encoder_type: Type of encoding ('onehot', 'label', or 'ordinal')
            keep_original: Whether to keep original columns
            
        Returns:
            DataFrame with encoded columns
        """
        try:
            #get and print categorical columns
            print('\nDATA PROCESS: categorical columns to encode: \n', df.select_dtypes(exclude=[np.number]).columns)

            # Get columns to encode based on given column named  in column_indices varaibale
            columns_to_encode = column_indices
            print('\nDATA PROCESS: columns_to_encode: \n', columns_to_encode)
            print('\nDATA PROCESS: columns_to_encode type: \n', type(columns_to_encode))
            
            self.logger.info(f"Encoding columns: {columns_to_encode}")
            
            # Initialize encoder
            if encoder_type == 'onehot':
                
                # Fit and transform
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                
                encoded_data = encoder.fit_transform(df[columns_to_encode].values)
                
                # Get feature names for one-hot encoded columns
                feature_names = encoder.get_feature_names_out(columns_to_encode)
                #print(f'feature_names after onehot encode for columns \n {df.columns[columns_to_encode]}: ', feature_names)
                
                # Create DataFrame with encoded data
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=feature_names,
                    index=df.index
                )
                
                print('DATA PROCESS: encoded_df after onehot encode for columns \n : ', encoded_df)
                
            elif encoder_type == 'label':
                encoded_df = pd.DataFrame(index=df.index)
                
                # Handle each column separately for label encoding
                for col in columns_to_encode:
                    encoder = LabelEncoder()
                    encoded_data = encoder.fit_transform(df[col].astype(str))
                    encoded_df[f"{col}_categorical_encoded"] = encoded_data
                    self.encoders[col] = encoder

                print('DATA PROCESS: encoded_df after label encode for columns \n : ', encoded_df)
            elif encoder_type == 'ordinal':
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                
                # Fit and transform
                encoded_data = encoder.fit_transform(df[columns_to_encode])
                
                # Create DataFrame with encoded data
                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=[f"{col}_categorical_encoded" for col in columns_to_encode],
                    index=df.index
                )
                
                print('DATA PROCESS: encoded_df after ordinal encode for columns \n : ', encoded_df)
                self.encoders['ordinal'] = encoder
                
            else:
                raise ValueError(f"Unsupported encoder type: {encoder_type}")
            
            # Store encoder for future use
            if encoder_type == 'onehot':
                self.encoders['onehot'] = encoder
            
            # Remove original columns if not keeping them
            if not keep_original:
                df = df.drop(columns=columns_to_encode)
            
            # Combine encoded columns with remaining columns
            df = pd.concat([df, encoded_df], axis=1)
            
            self.logger.info(f"DATA PROCESS: Encoding completed. New shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error during encoding: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, data_file_name) -> None: # save the processed data to a file in folder data/processed
        
        print('\nDATA PROCESS: Saving processed data to a file\n')
        processed_data_path = self.config['data']['processed_data_path']

        if data_file_name.endswith('.csv'):
            df.to_csv(f'{processed_data_path}processed_{data_file_name}', index=False)

        elif data_file_name.endswith('.xlsx'):
            df.to_excel(f'{processed_data_path}processed_{data_file_name}', index=False)

        else:
            df.to_csv(f'{processed_data_path}processed_{data_file_name}.csv', index=False)             


    def vectorize_text_columns(self, 
                           df: pd.DataFrame, 
                           text_columns: List[str], 
                           keep_original: bool = False) -> pd.DataFrame:
        """
        Traite les colonnes textuelles spécifiées d'un DataFrame en les vectorisant. Puis ajoute les nouvelles features au DataFrame.
        
        Args:
            df: Le DataFrame contenant les données
            text_columns: Liste des noms de colonnes à traiter
            keep_original: Si True, conserve les colonnes originales
        
        Returns:
            pd.DataFrame: DataFrame avec les nouvelles features textuelles vectorisées
        """
        if not text_columns:
            #self.logger.info("Aucune colonne textuelle à traiter.")
            print("DATA PROCESS (vectorization): Aucune colonne textuelle à traiter.")
            return df
          
        # Initialisation des outils NLTK (pour le prétraitement de texte)
        try:
            self.logger.info("Téléchargement des ressources NLTK nécessaires...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
        except Exception as e:
            self.logger.warning(f"Erreur lors du téléchargement NLTK: {e}")
            self.logger.info("Tentative de continuer avec les ressources disponibles...")
            
        # Vérification de l'existence des colonnes
        missing_cols = [col for col in text_columns if col not in df.columns]
        if missing_cols:
            print(f"DATA PROCESS (vectorization): Colonnes non trouvées dans le DataFrame: {missing_cols}")
            raise ValueError(f"Colonnes non trouvées dans le DataFrame: {missing_cols}")
            
        text_features_config = self.config.get('features', {}).get('text_features', {})
        max_features = text_features_config.get('max_features', 100) # up to 100 features (words)
        min_df = text_features_config.get('min_df', 0.01)            # Remove rare words
        max_df = text_features_config.get('max_df', 0.95)            # Remove too common words
        ngram_range = text_features_config.get('ngram_range', (1, 2)) # Include pairs of words
          
        #Parameters
        self.lemmatizer = WordNetLemmatizer()    
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
                        
        # Copie du DataFrame pour éviter la modification en place
        result_df = df.copy()
        
        start_time = time.time()
        total_features = 0
        
        for col in text_columns:
            #self.logger.info(f"\nTraitement de la colonne '{col}'...")
            print(f"\nDATA PROCESS (vectorization): Traitement de la colonne '{col}'...")
            
            # Prétraitement
            col_process_start = time.time()
            #self.logger.info("Étape 1: Nettoyage et normalisation du texte...")
            print("\nDATA PROCESS (vectorization): Étape 1: Nettoyage et normalisation du texte...")
            
            processed_col_name = f"{col}_processed"
            result_df[processed_col_name] = df[col].apply(self._clean_text)
            
            # Vectorisation
            #self.logger.info("Étape 2: Vectorisation du texte...")
            print("\nDATA PROCESS (vectorization): Étape 2: Vectorisation du texte...")
            sparse_features = self.vectorizer.fit_transform(result_df[processed_col_name])
            
            # Création des nouvelles colonnes
            feature_names = self.vectorizer.get_feature_names_out()
            vectorized_df = pd.DataFrame(
                sparse_features.toarray(),
                columns=[f"{col}_feature_{word}" for word in feature_names]
            )
            
            # Ajout des nouvelles colonnes au DataFrame
            result_df = pd.concat([result_df, vectorized_df], axis=1)
            
            # Suppression de la colonne de texte prétraité
            result_df.drop(columns=[processed_col_name], inplace=True)
            
            # Suppression de la colonne originale si demandé
            if not keep_original:
                result_df.drop(columns=[col], inplace=True)
            
            # Statistiques de traitement
            process_time = time.time() - col_process_start
            features_added = len(feature_names)
            total_features += features_added
            
            # self.logger.info(f"Traitement de '{col}' terminé:")
            # self.logger.info(f"- Temps de traitement: {process_time:.2f} secondes")
            # self.logger.info(f"- Nombre de features créées: {features_added}")
            # self.logger.info(f"- Exemples de features: {feature_names[:5]}...")
            print(f"DATA PROCESS (vectorization): Traitement de '{col}' terminé:")
            print(f"- Temps de traitement: {process_time:.2f} secondes")
            print(f"- Nombre de features créées: {features_added}")
            print(f"- Exemples de features: {feature_names[:5]}...")

            # concerver le vocabulaire de la vectorisation pour une une analyse du clustering ou autre
            self.vectorizer_vocabulary_ = feature_names
         

        # Statistiques finales
        # total_time = time.time() - start_time
        # self.logger.info(f"\nTraitement terminé!")
        # self.logger.info(f"Temps total: {total_time:.2f} secondes")
        # self.logger.info(f"Total des features créées: {total_features}")
        # self.logger.info(f"Dimensions finales du DataFrame: {result_df.shape}")
        print(f"\nDATA PROCESS (vectorization): Traitement terminé!")
        # print(f"Temps total: {total_time:.2f} secondes")
        print(f"Total des features créées: {total_features}")
        print(f"Dimensions finales du DataFrame: {result_df.shape}")
        
        return result_df
    
    
    def _clean_text(self, text) -> str: #Nettoyer et normaliser un texte donné (adaptation forcé en chaine de caractère , conversion en minuscules, Tokenization (découpage en mots), lemmatization (garder uniquement les mots nécessaires), etc..)
        """
        Sera appeler en premier par les fonctions qui nécessittent de vectoriser une colonne texte avant le traitement de vectorisation de la colonne
        Nettoyer et normaliser un texte donné (adaptation forcé en chaine de caractère , conversion en minuscules, Tokenization (découpage en mots), lemmatization (garder uniquement les mots nécessaires), etc..)
        
        Args:
            text: Le texte à nettoyer
        Returns:
            str: Le texte nettoyé
        """
        # Gestion des valeurs non textuelles
        if pd.isna(text):
            return ""
        
        try:
            # Conversion forcée en string
            text = str(text)
            
            # Conversion en minuscules
            text = text.lower()
            
            # Tokenization
            tokens = word_tokenize(text)
            
            # Suppression des stopwords et lemmatisation
            cleaned_tokens = []
            for token in tokens:
                if (token.isalnum() and  # Uniquement les caractères alphanumériques
                    token not in self.stop_words and  # Pas de stopwords
                    len(token) > 1):  # Évite les caractères isolés
                    lemmatized = self.lemmatizer.lemmatize(token)
                    cleaned_tokens.append(lemmatized)
            
            return ' '.join(cleaned_tokens)

        except Exception as e:
            #self.logger.error(f"Erreur lors du nettoyage du texte: {e}")
            print(f"\nDATA PROCESS (cleaning and adapting text features): Erreur lors du nettoyage du texte: {e}")
            return ""
        
    def get_data_frame(self) -> pd.DataFrame:
        """Return the DataFrame of the instance (alerady load fom file)."""
        return self.df

    def scale_features(self, features: List[str]) -> pd.DataFrame:
        """Scale numeric features."""
        X = self.df[features]
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=features)

    def encode_features(self, features: List[str]) -> pd.DataFrame:
        """Encode categorical features."""
        X = self.df[features]
        X_encoded = self.encoder.fit_transform(X)
        return pd.DataFrame(X_encoded.toarray(), columns=self.encoder.get_feature_names(features))    

class FeatureStats: #Object used to store each features statistics info. used by the DataAnalyzer class
    
    def __init__(self, name: str, dtype: str, missing_values_count: int, unique_values_count: int, numeric_stats: Optional[Dict] = None, categorical_stats: Optional[Dict] = None):
        """
        Initialize the metadata object.
        
        Args:
            name: Name of the feature
            dtype: Data type of the feature
            missing_values_count: Number of missing values
            unique_values_count: Number of unique values
            numeric_stats: Dictionary of numeric feature statistics
            categorical_stats: Dictionary of categorical feature statistics
        """
        self.name = name
        self.dtype = dtype
        self.missing_values_count = missing_values_count
        self.unique_values_count = unique_values_count
        self.numeric_stats = numeric_stats
        self.categorical_stats = categorical_stats

    # name: str
    # dtype: str
    # missing_values_count: int 
    # unique_values_count: int
    # numeric_stats: Optional[Dict] = None
    # categorical_stats: Optional[Dict] = None
       
class DataAnalyzer:
    """Class for automated exploratory data analysis."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.

        Args:
            df: Input DataFrame to analyze
        """
        self.df = df
        self.features: List[FeatureMetadata] = []
        self.feature_stats: Dict [str, FeatureStats] = {}
        self._analyze_features()

    def _analyze_features(self) -> None:
        """Analyze all features in the DataFrame."""
        
        for column in self.df.columns:
            missing_valeus_count = self.df[column].isnull().sum()
            unique_values_count = self.df[column].nunique()
            
            #if self.df[columns].dtype == np.number:
            if pd.api.types.is_numeric_dtype(self.df[column]):
                numeric_stats = {
                    'mean':  self.df[column].mean(),
                    'std': self.df[column].std(),
                    'min': self.df[column].min(),
                    'max': self.df[column].max(),
                    'skew': self.df[column].skew(),
                    'kurtosis': self.df[column].kurtosis(),
                    'median': self.df[column].median()
                }
                featureStat = FeatureStats(column, 'numeric', missing_valeus_count, unique_values_count, numeric_stats)
                self.feature_stats[column] = featureStat

            else: # This section for categorical features will never be reach if we already processsed the data with the DataProcessor (makin sure all categorical features are encoded)
                categorical_stats = {
                    'mode': self.df[column].mode()[0],
                    'entropy': self._calculate_entropy(self.df[column].value_counts())
                }
                featureStat = FeatureStats(column, 'categorical', missing_valeus_count, unique_values_count, None, categorical_stats)
                self.feature_stats[column] = featureStat
        
        print('\nDATA ANALYSE: number of features analysed for statistics', len(self.feature_stats))

    def generate_summary(self) -> Dict:
        """Generate a summary of the dataset."""
        
        print('\nDATA ANALYSE: generating summary\n')
        return {
            'dataset_shape': self.df.shape,
            'total_missing': self.df.isnull().sum().sum(),
            'numeric_features': len([f for f in self.feature_stats.values() 
                                   if f.dtype == 'numeric']),
            'categorical_features': len([f for f in self.feature_stats.values() 
                                       if f.dtype == 'categorical']),
            'feature_stats': {name: vars(stats) 
                            for name, stats in self.feature_stats.items()}
        }

    def plot_correlations(self, save_path: Optional[str] = None) -> None: # to visualize the correlation between the features
        """Plot correlation matrix for numeric features."""
        
        features_names = self.df.columns  # We assumed that all features are already encoded and scaled. So we can use all the features for the correlation matrix
        
        # If there are too many features (more thant 15), limit the number of features to plot to 10 randomly
        if len(features_names) > 15:
            print('\nDATA ANALYSE: Too many features to plot correlation. Limiting to 10 randomly\n')
            features_names = np.random.choice(features_names, 10, replace=False)

        print('\nDATA ANALYSE: features_names to plot correlation: \n', features_names)

        corr_matrix = self.df[features_names].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('\n DATA ANALYSE: Feature Correlations')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    @staticmethod
    def _calculate_entropy(value_counts: pd.Series) -> float:
        """Calculate entropy of a categorical variable."""
        proportions = value_counts / value_counts.sum()
        return -np.sum(proportions * np.log2(proportions))
    
    def plot_distributions(self, max_features: int = 5) -> None:
        """Plot distributions of numeric features."""
        numeric_features = [f.name for f in self.features 
                          if f.dtype == 'numeric'][:max_features]

        fig, axes = plt.subplots(len(numeric_features), 2, 
                                figsize=(12, 4*len(numeric_features)))

        for i, feature in enumerate(numeric_features):
            # Histogram
            sns.histplot(data=self.df, x=feature, ax=axes[i, 0])
            axes[i, 0].set_title(f'{feature} Distribution')
            
            # Box plot
            sns.boxplot(data=self.df, x=feature, ax=axes[i, 1])
            axes[i, 1].set_title(f'{feature} Box Plot')

        plt.tight_layout()
        plt.show()


    def analyze_pca_components(self, n_components: int = 2) -> Tuple:
        """Perform PCA analysis on numeric features."""
        numeric_features = [f.name for f in self.features 
                          if f.dtype == 'numeric']

        if len(numeric_features) < 2:
            raise ValueError("Need at least 2 numeric features for PCA")

        # Prepare data
        X = self.df[numeric_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Calculate explained variance
        explained_variance = pca.explained_variance_ratio_

        return X_pca, explained_variance, pca.components_

    def suggest_preprocessing(self) -> Dict:
        """Suggest preprocessing steps for features."""
        suggestions = {
            'scaling_needed': [],
            'encoding_needed': [],
            'missing_values': [],
            'skewed_features': []
        }

        for feature in self.features:
            if feature.missing_count > 0:
                suggestions['missing_values'].append(feature.name)

            if feature.dtype == 'numeric':
                if feature.numeric_stats['skew'] > 1:
                    suggestions['skewed_features'].append(feature.name)
                if (feature.numeric_stats['max'] - 
                    feature.numeric_stats['min']) > 10:
                    suggestions['scaling_needed'].append(feature.name)

            if feature.dtype == 'categorical':
                suggestions['encoding_needed'].append(feature.name)

        return suggestions