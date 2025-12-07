"""
Módulo para entrenamiento y evaluación de modelos ML
Incluye tracking con MLflow y experimentación
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """Clase para entrenamiento y evaluación de modelos WIDS 2024"""
    
    def __init__(self, experiment_name="wids2024_experiment"):
        """
        Inicializa el entrenador de modelos
        
        Args:
            experiment_name (str): Nombre del experimento en MLflow
        """
        self.experiment_name = experiment_name
        self.best_model = None
        self.best_params = None
        self.best_score = None
        
        # Configurar MLflow
        mlflow.set_experiment(self.experiment_name)
    
    def auc_scoring(self, estimator, X, y):
        """
        Función de scoring personalizada para AUC
        Usada en GridSearchCV
        
        Args:
            estimator: Modelo a evaluar
            X: Características
            y: Etiquetas
            
        Returns:
            float: Score AUC
        """
        predictions = estimator.predict_proba(X)[:, 1]
        return roc_auc_score(y, predictions)
    
    def train_with_gridsearch(self, X_train, y_train, cv_splits=5):
        """
        Entrena modelo con GridSearchCV usando los mismos parámetros del notebook
        
        Args:
            X_train (pd.DataFrame): Características de entrenamiento
            y_train (pd.Series): Etiquetas de entrenamiento
            cv_splits (int): Número de folds para cross-validation
            
        Returns:
            tuple: (best_model, best_params, best_score)
        """
        # Definir parámetros igual que en el notebook
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0],
            'algorithm': ['SAMME'],
            'estimator': [DecisionTreeClassifier(max_depth=3)]
        }
        
        # Configurar cross-validation estratificada
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        # Crear GridSearchCV
        grid_search = GridSearchCV(
            AdaBoostClassifier(),
            param_grid,
            cv=cv,
            scoring=self.auc_scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Iniciar run de MLflow
        with mlflow.start_run(run_name="ada_boost_gridsearch"):
            # Entrenar
            print("🔍 Ejecutando GridSearchCV...")
            grid_search.fit(X_train, y_train)
            
            # Guardar resultados
            self.best_model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
            
            # Loggear en MLflow
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_cv_auc", self.best_score)
            mlflow.sklearn.log_model(self.best_model, "best_ada_boost_model")
            
            # Loggear otros parámetros
            mlflow.log_param("model_type", "AdaBoost")
            mlflow.log_param("base_estimator", "DecisionTree(max_depth=3)")
            mlflow.log_param("cv_folds", cv_splits)
            mlflow.log_param("features_used", list(X_train.columns))
        
        print(f"✅ Mejores parámetros: {self.best_params}")
        print(f"✅ Mejor score AUC: {self.best_score:.4f}")
        
        return self.best_model, self.best_params, self.best_score
    
    def evaluate_model(self, model, X_val, y_val):
        """
        Evalúa el modelo en conjunto de validación
        
        Args:
            model: Modelo entrenado
            X_val (pd.DataFrame): Características de validación
            y_val (pd.Series): Etiquetas de validación
            
        Returns:
            dict: Métricas de evaluación
        """
        # Predecir probabilidades
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Calcular métricas
        auc_score = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Loggear en MLflow si hay un run activo
        try:
            mlflow.log_metric("validation_auc", auc_score)
            mlflow.log_metric("validation_accuracy", accuracy)
        except:
            pass  # No hay run activo de MLflow
        
        metrics = {
            'auc': auc_score,
            'accuracy': accuracy,
            'classification_report': classification_report(y_val, y_pred)
        }
        
        print(f"📊 AUC en validación: {auc_score:.4f}")
        print(f"📊 Accuracy en validación: {accuracy:.4f}")
        
        return metrics
    
    def predict_test_set(self, model, X_test, return_proba=True):
        """
        Genera predicciones para el conjunto de test
        
        Args:
            model: Modelo entrenado
            X_test (pd.DataFrame): Características de test
            return_proba (bool): Si True retorna probabilidades, si False clases
            
        Returns:
            np.array: Predicciones
        """
        if return_proba:
            predictions = model.predict_proba(X_test)[:, 1]
        else:
            predictions = model.predict(X_test)
        
        print(f"✅ Predicciones generadas: {len(predictions)} muestras")
        print(f"   Rango: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        return predictions
    
    def create_submission(self, predictions, test_df, sample_submission_path='data/sample_submission.csv'):
        """
        Crea archivo de submission en formato CSV
        
        Args:
            predictions (np.array): Predicciones del modelo
            test_df (pd.DataFrame): DataFrame de test original
            sample_submission_path (str): Ruta al archivo de submission de ejemplo
            
        Returns:
            pd.DataFrame: DataFrame de submission
        """
        # Cargar template de submission
        submission = pd.read_csv(sample_submission_path)
        
        # Asegurar que tenemos suficientes predicciones
        if len(predictions) < len(submission):
            raise ValueError(f"No hay suficientes predicciones: {len(predictions)} < {len(submission)}")
        
        # Asignar predicciones
        submission['DiagPeriodL90D'] = predictions[:len(submission)]
        
        # Guardar archivo
        output_path = 'submission_final.csv'
        submission.to_csv(output_path, index=False)
        
        print(f"✅ Submission guardada en: {output_path}")
        print(f"   Tamaño: {len(submission)} filas")
        print(f"   Rango de predicciones: [{submission['DiagPeriodL90D'].min():.4f}, "
              f"{submission['DiagPeriodL90D'].max():.4f}]")
        
        # Loggear en MLflow si hay run activo
        try:
            mlflow.log_artifact(output_path)
        except:
            pass
        
        return submission