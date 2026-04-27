"""
Neural Network Model 2 - Two-Phase Optimization
=================================================
Advanced neural network with a two-phase Optuna optimization strategy:
Phase 1: Broad search (250 trials) | Phase 2: Local refinement (150 trials).

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
from pathlib import Path
from typing import Dict, Any

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# PATH CONFIGURATION - EDIT THIS SECTION
# =============================================================================
BASE_PATH = Path('/content')
TRAIN_PATH = BASE_PATH / 'train_balanced_stratified.csv'
VAL_PATH = BASE_PATH / 'validation_balanced_stratified.csv'
OUTPUT_DIR = BASE_PATH / 'model_output_two_phase'
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1. DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data():
    """Loads and preprocesses training and external validation data."""
    print("\nLOADING DATA...")

    df_train = pd.read_csv(TRAIN_PATH)
    df_val_final = pd.read_csv(VAL_PATH)

    print(f"Training set: {df_train.shape[0]} samples, {df_train.shape[1]} features")
    print(f"External validation: {df_val_final.shape[0]} samples")

    comp_col = 'composition' if 'composition' in df_train.columns else 'final_composition'
    gap_col = 'gap expt'
    magpie_columns = [col for col in df_train.columns if 'MagpieData' in col]

    X_train = df_train[magpie_columns].values
    y_train = df_train[gap_col].values
    X_val_final = df_val_final[magpie_columns].values
    y_val_final = df_val_final[gap_col].values

    print(f"Magpie features: {len(magpie_columns)}")

    # Robust scaling for features
    scaler_features = RobustScaler(quantile_range=(5, 95))
    X_train_scaled = scaler_features.fit_transform(X_train)
    X_val_final_scaled = scaler_features.transform(X_val_final)

    # Target scaling to [0, 1] for softplus
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    y_train_scaled = scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_final_scaled = scaler_target.transform(y_val_final.reshape(-1, 1)).flatten()

    print("Preprocessing completed")
    return {
        'X_train': X_train_scaled,
        'y_train': y_train_scaled,
        'df_train': df_train,
        'X_val_final': X_val_final_scaled,
        'y_val_final': y_val_final_scaled,
        'df_val_final': df_val_final,
        'scaler_features': scaler_features,
        'scaler_target': scaler_target,
        'magpie_columns': magpie_columns,
        'comp_col': comp_col,
        'gap_col': gap_col
    }

# =============================================================================
# 2. MODEL CREATION
# =============================================================================

def create_model(params: Dict[str, Any], input_dim: int):
    """Creates the model with the specified parameters."""
    n_layers = params['n_layers']
    dropout_rate = params['dropout_rate']
    activation = params['activation']
    use_batch_norm = params['use_batch_norm']
    use_residual = params['use_residual']
    l2_reg = params['l2_reg']

    inputs = layers.Input(shape=(input_dim,))
    x = inputs

    for i in range(n_layers):
        units = params[f'n_units_layer_{i}']

        x_new = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            kernel_initializer='he_normal'
        )(x)

        if use_batch_norm:
            x_new = layers.BatchNormalization()(x_new)

        if use_residual and i > 0 and x.shape[-1] == units:
            x_new = layers.Add()([x, x_new])

        x = x_new
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='softplus')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer
    lr = params['learning_rate']
    optimizer_name = params['optimizer']

    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=lr)
    elif optimizer_name == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=l2_reg)
    else:
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    # Loss function
    loss_name = params['loss_function']
    if loss_name == 'mae':
        loss = 'mae'
    elif loss_name == 'mse':
        loss = 'mse'
    elif loss_name == 'huber':
        loss = keras.losses.Huber(delta=1.0)
    else:
        loss = keras.losses.LogCosh()

    model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
    return model

# =============================================================================
# 3. COMMON TRAINING FUNCTION
# =============================================================================

def train_model(params: Dict[str, Any], data: Dict[str, Any],
                prune_callback=None, epochs: int = 300) -> float:
    """Trains a model and returns the best val_mae."""
    model = create_model(params, data['X_train'].shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=40,
            restore_best_weights=True,
            min_delta=0.0005,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=20,
            min_lr=1e-8,
            mode='min'
        )
    ]

    if prune_callback:
        callbacks.append(prune_callback)

    history = model.fit(
        data['X_train'], data['y_train'],
        batch_size=params['batch_size'],
        epochs=epochs,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
        shuffle=True
    )

    return min(history.history['val_mae'])

# =============================================================================
# 4. OPTUNA OBJECTIVE FUNCTIONS
# =============================================================================

def create_objective_phase1(data: Dict[str, Any]):
    """Creates the objective function for Phase 1 (broad search)."""
    def objective(trial):
        params = {
            'n_layers': trial.suggest_int('n_layers', 3, 12),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
            'learning_rate': trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'swish']),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'nadam', 'adamw']),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'l2_reg': trial.suggest_float('l2_reg', 1e-8, 1e-2, log=True),
            'loss_function': trial.suggest_categorical('loss_function', ['mae', 'mse', 'huber']),
        }

        for i in range(params['n_layers']):
            params[f'n_units_layer_{i}'] = trial.suggest_int(f'n_units_layer_{i}', 8, 1024)

        return train_model(params, data, TFKerasPruningCallback(trial, 'val_mae'))

    return objective


def create_objective_phase2(data: Dict[str, Any], base_params: Dict[str, Any]):
    """Creates the objective function for Phase 2 (refinement)."""
    def objective(trial):
        # Restricted search space
        n_layers_base = base_params['n_layers']
        lr_base = base_params['learning_rate']

        params = {
            'n_layers': trial.suggest_int('n_layers', max(2, n_layers_base-1), min(12, n_layers_base+2)),
            'dropout_rate': trial.suggest_float('dropout_rate',
                                               max(0.05, base_params['dropout_rate']-0.2),
                                               min(0.8, base_params['dropout_rate']+0.2)),
            'learning_rate': trial.suggest_float('learning_rate',
                                                lr_base/5, lr_base*5, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'activation': trial.suggest_categorical('activation', [base_params['activation']]),
            'optimizer': trial.suggest_categorical('optimizer', [base_params['optimizer']]),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [base_params['use_batch_norm']]),
            'use_residual': trial.suggest_categorical('use_residual', [base_params['use_residual']]),
            'l2_reg': trial.suggest_float('l2_reg',
                                         max(1e-9, base_params['l2_reg']/10),
                                         min(1e-1, base_params['l2_reg']*10), log=True),
            'loss_function': trial.suggest_categorical('loss_function', [base_params['loss_function']]),
        }

        # For layers, use restricted range around base values
        for i in range(params['n_layers']):
            if f'n_units_layer_{i}' in base_params:
                base_units = base_params[f'n_units_layer_{i}']
                min_units = max(8, int(base_units * 0.5))
                max_units = min(1024, int(base_units * 1.5))
                params[f'n_units_layer_{i}'] = trial.suggest_int(f'n_units_layer_{i}', min_units, max_units)
            else:
                # Additional layers if n_layers increased
                params[f'n_units_layer_{i}'] = trial.suggest_int(f'n_units_layer_{i}', 8, 512)

        # Phase 2: Longer training for refinement
        return train_model(params, data, TFKerasPruningCallback(trial, 'val_mae'), epochs=400)

    return objective

# =============================================================================
# 5. CROSS-VALIDATION
# =============================================================================

def cross_validation(params: Dict[str, Any], data: Dict[str, Any]):
    """Runs 10-fold cross validation."""
    print("\nSTARTING CROSS-VALIDATION (10-FOLD)...")

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data['X_train'])):
        print(f"   Fold {fold + 1}/10...", end=' ')

        X_tr, X_val = data['X_train'][train_idx], data['X_train'][val_idx]
        y_tr, y_val = data['y_train'][train_idx], data['y_train'][val_idx]

        model = create_model(params, data['X_train'].shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_mae',
                patience=50,
                restore_best_weights=True,
                min_delta=0.0005
            )
        ]

        history = model.fit(
            X_tr, y_tr,
            batch_size=params['batch_size'],
            epochs=500,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )

        val_mae = model.evaluate(X_val, y_val, verbose=0)[1]
        cv_scores.append(val_mae)
        print(f"MAE: {val_mae:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCV completed: {cv_mean:.4f} +/- {cv_std:.4f} eV")
    return cv_scores, cv_mean, cv_std

# =============================================================================
# 6. EVALUATION AND ANALYSIS
# =============================================================================

def evaluate_on_validation_set(model, data):
    """Evaluates on the external validation set."""
    print("\nEVALUATING ON EXTERNAL VALIDATION SET...")

    y_pred_scaled = model.predict(data['X_val_final'], verbose=0).flatten()
    y_pred = data['scaler_target'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = data['scaler_target'].inverse_transform(data['y_val_final'].reshape(-1, 1)).flatten()

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100,
        'max_error': np.max(np.abs(y_true - y_pred))
    }

    if 'gap_category' in data['df_val_final'].columns:
        metrics['category'] = analyze_by_category(y_true, y_pred, data['df_val_final'], data['gap_col'])

    metrics['metallic'] = analyze_metallic_predictions(y_true, y_pred)
    metrics['outliers'] = analyze_outliers(y_true, y_pred)

    return y_pred, metrics


def analyze_by_category(y_true, y_pred, df, gap_col):
    analysis = {}
    df_temp = df.copy()
    df_temp['true'] = y_true
    df_temp['pred'] = y_pred
    df_temp['abs_error'] = np.abs(y_pred - y_true)

    for category in df_temp['gap_category'].unique():
        mask = df_temp['gap_category'] == category
        subset = df_temp[mask]

        analysis[category] = {
            'mae': mean_absolute_error(subset['true'], subset['pred']),
            'rmse': np.sqrt(mean_squared_error(subset['true'], subset['pred'])),
            'r2': r2_score(subset['true'], subset['pred']),
            'n_samples': len(subset),
            'error_std': subset['abs_error'].std()
        }

    return analysis


def analyze_metallic_predictions(y_true, y_pred):
    metallic_mask = y_true == 0
    if not np.any(metallic_mask):
        return None

    pred_metallic = y_pred[metallic_mask]
    return {
        'mean': pred_metallic.mean(),
        'std': pred_metallic.std(),
        'max': pred_metallic.max(),
        'above_0.1': np.sum(pred_metallic > 0.1),
        'above_0.5': np.sum(pred_metallic > 0.5)
    }


def analyze_outliers(y_true, y_pred, thresholds=[0.5, 1.0, 2.0]):
    error = np.abs(y_pred - y_true)
    return {f'>{t}eV': np.sum(error > t) for t in thresholds}

# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================

def create_visualizations(y_true, y_pred, metrics, cv_scores, output_dir):
    print("\nGENERATING VISUALIZATIONS...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    error = np.abs(y_true - y_pred)

    # 1. Predictions vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=15, c=error, cmap='viridis')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Experimental Band Gap (eV)')
    axes[0, 0].set_ylabel('Predicted Band Gap (eV)')
    axes[0, 0].set_title(f'Predictions vs Actual\nR2 = {metrics["r2"]:.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Absolute Error (eV)')

    # 2. Error distribution
    axes[0, 1].hist(error, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[0, 1].axvline(np.median(error), color='red', linestyle='--',
                      label=f'Median: {np.median(error):.3f} eV')
    axes[0, 1].set_xlabel('Absolute Error (eV)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Metrics by category
    if 'category' in metrics:
        categories = list(metrics['category'].keys())
        mae_values = [metrics['category'][cat]['mae'] for cat in categories]

        axes[1, 0].bar(categories, mae_values, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_ylabel('MAE (eV)')
        axes[1, 0].set_title('MAE by Category')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Cross-validation results
    axes[1, 1].plot(range(1, 11), cv_scores, 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 1].axhline(np.mean(cv_scores), color='red', linestyle='--',
                      label=f'Mean +/- Std: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f} eV')
    axes[1, 1].set_xlabel('Fold')
    axes[1, 1].set_ylabel('Validation MAE (eV)')
    axes[1, 1].set_title('10-Fold Cross-Validation Results')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Band gap distribution
    axes[2, 0].hist(y_true, bins=50, alpha=0.7, label='Actual', density=True, edgecolor='black')
    axes[2, 0].hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True, edgecolor='black')
    axes[2, 0].set_xlabel('Band Gap (eV)')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].set_title('Band Gap Distribution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Errors vs Actual values
    axes[2, 1].scatter(y_true, error, alpha=0.6, s=15)
    axes[2, 1].set_xlabel('Experimental Band Gap (eV)')
    axes[2, 1].set_ylabel('Absolute Error (eV)')
    axes[2, 1].set_title('Error vs Actual Value')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'advanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =============================================================================
# 8. MAIN FUNCTION WITH 2 PHASES
# =============================================================================

def main():
    print("=" * 80)
    print("ADVANCED NEURAL NETWORK - 2-PHASE OPTIMIZATION")
    print("PHASE 1: Broad search (250 trials) | PHASE 2: Refinement (150 trials)")
    print("=" * 80)

    # 1. Load data
    data = load_and_preprocess_data()

    # ==================== PHASE 1: BROAD SEARCH ====================
    print("\nPHASE 1 - BROAD SEARCH (250 trials)...")
    study_phase1 = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    objective_phase1 = create_objective_phase1(data)
    study_phase1.optimize(objective_phase1, n_trials=250, timeout=3600*8)

    print(f"\nPHASE 1 completed")
    print(f"Best MAE: {study_phase1.best_value:.4f}")
    print(f"Best parameters: {study_phase1.best_params}")

    # ==================== PHASE 2: REFINEMENT ====================
    print("\nPHASE 2 - LOCAL REFINEMENT (150 trials)...")
    print(f"Centered on: {study_phase1.best_params}")

    study_phase2 = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=42)  # More exploitation
    )

    objective_phase2 = create_objective_phase2(data, study_phase1.best_params)
    study_phase2.optimize(objective_phase2, n_trials=150, timeout=3600*6)

    # Choose the best between both phases
    if study_phase2.best_value < study_phase1.best_value:
        best_params = study_phase2.best_params
        best_value = study_phase2.best_value
        print(f"\nPHASE 2 improved the result to {best_value:.4f}")
    else:
        best_params = study_phase1.best_params
        best_value = study_phase1.best_value
        print(f"\nPHASE 1 kept the best result: {best_value:.4f}")

    # ==================== CROSS-VALIDATION ====================
    print("\nCROSS-VALIDATION WITH BEST PARAMETERS...")
    cv_scores, cv_mean, cv_std = cross_validation(best_params, data)

    # ==================== FINAL TRAINING ====================
    print("\nFINAL TRAINING WITH ALL DATA...")
    final_model = create_model(best_params, data['X_train'].shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='mae',
            patience=80,
            restore_best_weights=True,
            min_delta=0.0001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='mae',
            factor=0.2,
            patience=40,
            min_lr=1e-10
        )
    ]

    final_history = final_model.fit(
        data['X_train'], data['y_train'],
        batch_size=best_params['batch_size'],
        epochs=1200,
        callbacks=callbacks,
        verbose=0
    )

    # ==================== FINAL EVALUATION ====================
    y_pred_final, metrics = evaluate_on_validation_set(final_model, data)

    # ==================== VISUALIZATIONS ====================
    create_visualizations(
        data['scaler_target'].inverse_transform(data['y_val_final'].reshape(-1, 1)).flatten(),
        y_pred_final, metrics, cv_scores, OUTPUT_DIR
    )

    # ==================== SAVE RESULTS ====================
    print("\nSAVING RESULTS...")

    final_model.save(OUTPUT_DIR / 'final_model.keras')
    joblib.dump(data['scaler_features'], OUTPUT_DIR / 'scaler_features.pkl')
    joblib.dump(data['scaler_target'], OUTPUT_DIR / 'scaler_target.pkl')

    results = {
        'optimization_phases': {
            'phase1_best_value': study_phase1.best_value,
            'phase2_best_value': study_phase2.best_value,
            'phase1_n_trials': len(study_phase1.trials),
            'phase2_n_trials': len(study_phase2.trials)
        },
        'best_hyperparameters': best_params,
        'cv_metrics': {
            'mean': cv_mean,
            'std': cv_std,
            'folds': cv_scores
        },
        'validation_metrics': metrics
    }

    with open(OUTPUT_DIR / 'complete_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    predictions_df = pd.DataFrame({
        'composition': data['df_val_final'][data['comp_col']],
        'true_bandgap': data['scaler_target'].inverse_transform(data['y_val_final'].reshape(-1, 1)).flatten(),
        'predicted_bandgap': y_pred_final,
        'abs_error': np.abs(y_pred_final - data['scaler_target'].inverse_transform(data['y_val_final'].reshape(-1, 1)).flatten())
    })
    predictions_df.to_csv(OUTPUT_DIR / 'external_validation_predictions.csv', index=False)

    # ==================== FINAL SUMMARY ====================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - 2-PHASE OPTIMIZATION COMPLETED")
    print("=" * 80)
    print(f"EXTERNAL SET METRICS:")
    print(f"   - MAE: {metrics['mae']:.4f} eV")
    print(f"   - RMSE: {metrics['rmse']:.4f} eV")
    print(f"   - R2: {metrics['r2']:.4f}")
    print(f"   - CV Score: {cv_mean:.4f} +/- {cv_std:.4f} eV")
    print(f"\nFINAL ARCHITECTURE:")
    print(f"   - Layers: {best_params['n_layers']}")
    print(f"   - Neurons: {[best_params[f'n_units_layer_{i}'] for i in range(best_params['n_layers'])]}")
    print(f"   - Activation: {best_params['activation']}")
    print(f"   - Optimizer: {best_params['optimizer']}")
    print(f"\nFiles saved in: {OUTPUT_DIR}")
    print("=" * 80)

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == '__main__':
    main()
