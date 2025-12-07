"""
Módulo para carga y preprocesamiento de datos WIDS 2024
Automatiza el procesamiento del dataset de cáncer
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

class DataProcessor:
    """Clase para carga y preprocesamiento de datos WIDS 2024"""
    
    def __init__(self, train_path='data/train.csv', test_path='data/test.csv'):
        """
        Inicializa el procesador de datos
        
        Args:
            train_path (str): Ruta al archivo train.csv
            test_path (str): Ruta al archivo test.csv
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
    def load_data(self):
        """
        Carga los datasets de entrenamiento y prueba
        
        Returns:
            tuple: (train_df, test_df) DataFrames cargados
        """
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        
        # Eliminar patient_id como en el notebook
        if 'patient_id' in self.train_df.columns:
            self.train_df.drop(columns=['patient_id'], inplace=True)
        if 'patient_id' in self.test_df.columns:
            self.test_df.drop(columns=['patient_id'], inplace=True)
            
        return self.train_df, self.test_df
    
    def impute_missing_values(self):
        """
        Imputa valores faltantes usando las mismas estrategias del notebook
        """
        if self.train_df is None or self.test_df is None:
            self.load_data()
        
        # Identificar columnas categóricas y numéricas
        categorical_columns = self.train_df.select_dtypes(include=['object']).columns
        numerical_cols = self.train_df.select_dtypes(exclude=['object']).columns
        
        # Excluir variable objetivo
        if 'DiagPeriodL90D' in categorical_columns:
            categorical_columns = categorical_columns.drop('DiagPeriodL90D')
        if 'DiagPeriodL90D' in numerical_cols:
            numerical_cols = numerical_cols.drop('DiagPeriodL90D')
        
        # Imputar categóricas con moda
        for col in categorical_columns:
            if col != 'DiagPeriodL90D':
                mode = self.train_df[col].mode()[0]
                self.train_df.loc[:, col] = self.train_df[col].fillna(mode)
                self.test_df.loc[:, col] = self.test_df[col].fillna(mode)
        
        # Imputar numéricas con mediana
        for col in numerical_cols:
            if col != 'DiagPeriodL90D':
                median_val = self.train_df[col].median()
                self.train_df.loc[:, col] = self.train_df[col].fillna(median_val)
                self.test_df.loc[:, col] = self.test_df[col].fillna(median_val)
        
        return self.train_df, self.test_df
    
    def encode_categorical_features(self, cols_to_encode=None):
        """
        Codifica variables categóricas usando OrdinalEncoder
        
        Args:
            cols_to_encode (list): Lista de columnas a codificar. 
                                  Si es None, usa las del notebook
        
        Returns:
            tuple: (train_df, test_df) DataFrames codificados
        """
        if self.train_df is None or self.test_df is None:
            self.load_data()
            self.impute_missing_values()
        
        # Columnas del notebook
        if cols_to_encode is None:
            categorical_columns = self.train_df.select_dtypes(include=['object']).columns
            cols_to_encode = categorical_columns.tolist() + ['patient_zip3']
        
        # Combinar train y test para el encoding consistente
        temp_test = self.test_df.copy()
        temp_test['DiagPeriodL90D'] = 2  # Marcador temporal
        df_combined = pd.concat([self.train_df, temp_test])
        
        # Aplicar encoding
        for col in cols_to_encode:
            if col in df_combined.columns:
                self.encoder.fit(df_combined[[col]])
                self.train_df[col] = self.encoder.transform(self.train_df[[col]])
                self.test_df[col] = self.encoder.transform(self.test_df[[col]])
        
        return self.train_df, self.test_df
    
    def prepare_features(self):
        """
        Prepara las características finales para el modelado
        Basado en las columnas seleccionadas en el notebook
        
        Returns:
            tuple: (X_train, y_train, X_test, feature_columns)
        """
        if self.train_df is None or self.test_df is None:
            self.load_data()
            self.impute_missing_values()
            self.encode_categorical_features()
        
        # Columnas seleccionadas en el notebook
        feature_columns = [
            'breast_cancer_diagnosis_code',
            'metastatic_cancer_diagnosis_code', 
            'patient_zip3',
            'patient_age',
            'payer_type',
            'patient_state',
            'breast_cancer_diagnosis_desc'
        ]
        
        # Separar características y objetivo
        X_train = self.train_df[feature_columns].copy()
        y_train = self.train_df['DiagPeriodL90D'].copy()
        X_test = self.test_df[feature_columns].copy()
        
        return X_train, y_train, X_test, feature_columns
    
    def get_processed_data(self):
        """
        Método principal que ejecuta todo el pipeline de procesamiento
        
        Returns:
            tuple: (X_train, y_train, X_test, feature_columns)
        """
        self.load_data()
        self.impute_missing_values()
        self.encode_categorical_features()
        return self.prepare_features()