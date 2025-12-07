#!/usr/bin/env python
"""
Script principal para pipeline WIDS 2024
Orquesta la carga, procesamiento, entrenamiento y predicción
"""

import sys
import argparse
from module_data import DataProcessor
from module_ml import ModelTrainer
import pandas as pd

def parse_arguments():
    """Parsear argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Pipeline WIDS 2024')
    parser.add_argument('--train_path', default='data/train.csv', 
                        help='Ruta al archivo train.csv')
    parser.add_argument('--test_path', default='data/test.csv', 
                        help='Ruta al archivo test.csv')
    parser.add_argument('--submission_path', default='data/sample_submission.csv',
                        help='Ruta al template de submission')
    parser.add_argument('--experiment_name', default='wids2024_experiment',
                        help='Nombre del experimento en MLflow')
    parser.add_argument('--cv_splits', type=int, default=5,
                        help='Número de folds para cross-validation')
    
    return parser.parse_args()

def main():
    """Función principal del pipeline"""
    print("=" * 60)
    print("       PIPELINE WIDS 2024 - CLASIFICACIÓN DE CÁNCER")
    print("=" * 60)
    
    # Parsear argumentos
    args = parse_arguments()
    
    try:
        # ===== 1. PROCESAMIENTO DE DATOS =====
        print("\n[1/4] 📊 PROCESANDO DATOS...")
        processor = DataProcessor(
            train_path=args.train_path,
            test_path=args.test_path
        )
        
        X_train, y_train, X_test, feature_cols = processor.get_processed_data()
        
        print(f"   • Train shape: {X_train.shape}")
        print(f"   • Test shape: {X_test.shape}")
        print(f"   • Features: {feature_cols}")
        print(f"   • Clases: {y_train.unique()}")
        
        # ===== 2. ENTRENAMIENTO DEL MODELO =====
        print("\n[2/4] 🤖 ENTRENANDO MODELO (GridSearchCV)...")
        trainer = ModelTrainer(experiment_name=args.experiment_name)
        
        best_model, best_params, best_score = trainer.train_with_gridsearch(
            X_train=X_train,
            y_train=y_train,
            cv_splits=args.cv_splits
        )
        
        print(f"   • Mejor modelo: AdaBoost")
        print(f"   • Mejor AUC (CV): {best_score:.4f}")
        print(f"   • Mejores parámetros: {best_params}")
        
        # ===== 3. PREDICCIÓN =====
        print("\n[3/4] 🔮 GENERANDO PREDICCIONES...")
        predictions = trainer.predict_test_set(
            model=best_model,
            X_test=X_test,
            return_proba=True
        )
        
        # ===== 4. CREAR SUBMISSION =====
        print("\n[4/4] 💾 CREANDO ARCHIVO DE SUBMISSION...")
        submission_df = trainer.create_submission(
            predictions=predictions,
            test_df=processor.test_df,
            sample_submission_path=args.submission_path
        )
        
        # ===== RESUMEN FINAL =====
        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        print(f"📁 Archivos generados:")
        print(f"   • submission_final.csv (listo para Kaggle)")
        print(f"\n📊 Resumen del modelo:")
        print(f"   • Features utilizadas: {len(feature_cols)}")
        print(f"   • Mejor AUC (CV): {best_score:.4f}")
        print(f"   • Predicciones: {len(predictions)} muestras")
        print(f"\n🔗 MLflow:")
        print(f"   • Experimento: {args.experiment_name}")
        print(f"   • Modelo guardado en tracking")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())