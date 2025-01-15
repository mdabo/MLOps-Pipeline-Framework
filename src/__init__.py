"""
Main package initialization for EDA project.
This file makes the src directory a Python package.
"""

# Version of the package
__version__ = '0.1.0'

# You can define what gets exported when someone does 'from src import *'
__all__ = ['DataAnalyzer', 'DataProcessor', 'FeatureMetadata', 'BaseModel', 'ModelTrainer', 'LearningType', 'ModelFactory', 'SupervisedClassifier', 'SupervisedRegressor', 'UnsupervisedClustering', 'UnsupervisedDimensionReduction']
        
import yaml
import os
from pathlib import Path
import pandas as pd
import numpy as np
from .data.data_processing import DataAnalyzer, DataProcessor
from .features.feature_engineering import FeatureMetadata
from .models.model_training import ModelTrainer
from .models.base_model import BaseModel
from .models.model_factory import LearningType, ModelFactory, SupervisedClassifier, SupervisedRegressor, UnsupervisedClustering, UnsupervisedDimensionReduction
from rich import print, traceback #print will override the built-in print function vith more details and clear output

traceback.install() # help to see the error in the terminal more clearly

if __name__ == "__main__":
    
    ### 1. ------ Load and adapt raw data (with DataProcessor) ------ ###
   
    data_processor = DataProcessor('configs/config.yml')
    
    raw_data_path = data_processor.config['data']['raw_data_path']
    raw_data_file_name = data_processor.config['data']['raw_data_file_name']
    
        # load data and change columns data type (if needed) to infer the data type
    raw_data_df = data_processor.data_load(f'{raw_data_path}{raw_data_file_name}') 
    
    print('MAIN: Initial raw_data \n', raw_data_df.head())
    print('/nMAIN: Initial raw data size \n', raw_data_df.shape)

    ### ----------------------------------------------------------- ###
    
    #ask user if continue the rest of the code or not
    continue_ = input('Do you want to continue? (yes/no): \n')
    if continue_ == 'no':
       print
       exit()
    
    ### 2. ------    Complete processing of all features      ------ ###
    """ handling missing values, outliers, scaling, vectorize text colums, encoding categorials features, etc.. """
    if raw_data_df is not None:   
        data_preprocessed_df = data_processor.process_data(raw_data_df)# process data (handling missing values, outliers, scaling, vectorize text colums, encoding categorials features, etc..)
        if data_preprocessed_df is not None:
            pass
        else:
            print('MAIN: Data preprocessing failed')

    else:
        print('MAIN: Data loading failed')    
        
    ### ----------------------------------------------------------- ###        
    

    ### 3. ------      Analyse features and give insighs     ------ ###
    # Analyze data and give some insights (with DataAnalyzer)
    data_analyser = DataAnalyzer(data_preprocessed_df)
    
    print ('\n Data summary after process and analyse: \n', data_analyser.generate_summary())
    
    #plot the correlation matrix
    #data_analyser.plot_correlations()    
    ### ----------------------------------------------------------- ###
    
    
    ### 4. -----------      Build and train ML model   ------------ ###
    
    # Train model (BaseModel and ModelTrainer)

    # --Prepare features and target for training if supervised learning else only features
    # The below code work only for supervised learning. To be adapt for unsupervised learning !!!
    target_column_name = data_processor.config['data']['raw_data_target_column']['name']
    target_column_type = data_processor.config['data']['raw_data_target_column']['type']

    if target_column_type == 'categorical':
        target_column_name = f'{target_column_name}_categorical_encoded' # The target column name after encoding (if it's categorical) in the data_processor.process_data() method 
    
    if target_column_name in data_preprocessed_df.columns:
        X = data_preprocessed_df.drop(columns=[target_column_name])
        y = data_preprocessed_df[target_column_name]
        
        # Encode target if it's categorical
            # -- Allready encoded in the data_processor.process_data() method
            
    else:
        X = data_preprocessed_df
        y = None
        
        
    # Initialize and train model
    trainer = ModelTrainer(data_processor.config)
    
    # Train model (it will automatically determine if it's supervised/unsupervised)
    metrics = trainer.train_model(X, y)
    print("\n Training metrics:\n", metrics)    
    
    # Evaluate model

    # Save model if needed