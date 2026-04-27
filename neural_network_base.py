"""
Neural Network Model 1 - Base Architecture
=============================================
Base neural network architecture for band gap prediction.
Includes Bayesian optimization with Optuna and rigorous 5-fold cross-validation.

NOTE: File paths are set to '/content/' for Google Colab compatibility.
Please update these paths according to your local environment.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("=" * 70)
print("NEURAL NETWORK FOR BAND GAP PREDICTION")
print("BASE ARCHITECTURE + BAYESIAN OPTIMIZATION + CROSS-VALIDATION")
print("=" * 70)

# Reproducibility configuration
tf.random.set_seed(42)
np.random.seed(42)

# 1. LOAD AND PREPROCESS DATA
print("\nLOADING AND PREPROCESSING DATA...")

# Load balanced datasets
df_train = pd.read_csv('/content/train_balanced_stratified.csv')
df_test = pd.read_csv('/content/validation_balanced_stratified.csv')

print(f"Training set: {df_train.shape}")
print(f"Test set: {df_test.shape}")

# Identify columns
comp_col = 'composition' if 'composition' in df_train.columns else 'final_composition'
gap_col = 'gap expt'

# Separate features and target
magpie_columns = [col for col in df_train.columns if 'MagpieData' in col]
X_train = df_train[magpie_columns].values
y_train = df_train[gap_col].values
X_test = df_test[magpie_columns].values
y_test = df_test[gap_col].values

print(f"Magpie features: {len(magpie_columns)}")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing completed")

# 2. DEFINE BASE ARCHITECTURE AND MODELING FUNCTIONS
print("\nDEFINING NEURAL NETWORK ARCHITECTURE...")


def create_model(trial=None):
    """Creates a neural network model with optimizable hyperparameters."""

    # Fixed hyperparameters (for trial=None) or optimizable
    if trial is not None:
        # Search space for Optuna
        n_layers = trial.suggest_int('n_layers', 3, 8)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        # Dynamic architecture
        layer_sizes = []
        for i in range(n_layers):
            layer_sizes.append(trial.suggest_int(f'n_units_layer_{i}', 32, 512))
    else:
        # Default values (base architecture)
        n_layers = 4
        dropout_rate = 0.3
        learning_rate = 1e-3
        batch_size = 32
        layer_sizes = [256, 128, 64, 32]

    # Build model
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_scaled.shape[1],)))

    # Hidden layers
    for i, units in enumerate(layer_sizes):
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    # Output layer (softplus ensures band gap >= 0)
    model.add(layers.Dense(1, activation='softplus'))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mae',  # Mean Absolute Error - robust to outliers
        metrics=['mae', 'mse']
    )

    return model, batch_size if trial is not None else batch_size


# 3. BAYESIAN OPTIMIZATION WITH OPTUNA
print("\nCONFIGURING BAYESIAN OPTIMIZATION...")


def objective(trial):
    """Objective function for Optuna."""

    # Create model with suggested hyperparameters
    model, batch_size = create_model(trial)

    # Callbacks
    callbacks = [
        TFKerasPruningCallback(trial, 'val_mae'),
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=15,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    ]

    # Train with validation split
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=batch_size,
        epochs=200,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    # Return the best validation MAE
    best_val_mae = min(history.history['val_mae'])
    return best_val_mae


# Run Bayesian optimization
print("STARTING BAYESIAN OPTIMIZATION (50 iterations)...")

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.HyperbandPruner()
)

study.optimize(objective, n_trials=50, timeout=3600 * 4)  # 4 hours maximum

print("Bayesian optimization completed")
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation MAE: {study.best_value:.4f}")

# 4. RIGOROUS CROSS-VALIDATION WITH BEST HYPERPARAMETERS
print("\nRUNNING CROSS-VALIDATION (5-FOLD)...")

# Configure K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
cv_histories = []

# Train on each fold
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    print(f"Processing Fold {fold + 1}/5...")

    # Split data
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Create model with best hyperparameters
    best_params = study.best_params
    model, batch_size = create_model(None)  # We will apply best params manually

    # Manually apply best hyperparameters
    if best_params:
        model = keras.Sequential()
        model.add(layers.Input(shape=(X_train_scaled.shape[1],)))

        for i in range(best_params['n_layers']):
            units = best_params[f'n_units_layer_{i}']
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(best_params['dropout_rate']))

        model.add(layers.Dense(1, activation='softplus'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
            loss='mae',
            metrics=['mae', 'mse']
        )
        batch_size = best_params['batch_size']

    # Callbacks for this fold
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
    ]

    # Train
    history = model.fit(
        X_fold_train, y_fold_train,
        batch_size=batch_size,
        epochs=200,
        validation_data=(X_fold_val, y_fold_val),
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate
    val_mae = model.evaluate(X_fold_val, y_fold_val, verbose=0)[1]  # index 1 is MAE
    cv_scores.append(val_mae)
    cv_histories.append(history)

    print(f"   Fold {fold + 1} - Validation MAE: {val_mae:.4f}")

# Cross-validation statistics
print(f"\nCROSS-VALIDATION RESULTS (5-FOLD):")
print(f"   Average MAE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"   Range: {np.min(cv_scores):.4f} - {np.max(cv_scores):.4f}")

# 5. FINAL MODEL TRAINING
print("\nTRAINING FINAL MODEL WITH ALL TRAINING DATA...")

# Create final model with best hyperparameters
best_params = study.best_params
final_model, final_batch_size = create_model(None)

# Manually apply best hyperparameters to the final model
if best_params:
    final_model = keras.Sequential()
    final_model.add(layers.Input(shape=(X_train_scaled.shape[1],)))

    for i in range(best_params['n_layers']):
        units = best_params[f'n_units_layer_{i}']
        final_model.add(layers.Dense(units, activation='relu'))
        final_model.add(layers.BatchNormalization())
        final_model.add(layers.Dropout(best_params['dropout_rate']))

    final_model.add(layers.Dense(1, activation='softplus'))

    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params['learning_rate']),
        loss='mae',
        metrics=['mae', 'mse']
    )
    final_batch_size = best_params['batch_size']

# Callbacks for final training
final_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=25,
        restore_best_weights=True,
        min_delta=0.001
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7
    )
]

# Train final model with all training data
final_history = final_model.fit(
    X_train_scaled, y_train,
    batch_size=final_batch_size,
    epochs=300,
    callbacks=final_callbacks,
    verbose=1
)

print("Final training completed")

# 6. EVALUATION ON EXTERNAL TEST SET
print("\nEVALUATING ON EXTERNAL TEST SET...")

# Predict on test set
y_pred = final_model.predict(X_test_scaled).flatten()

# Calculate metrics
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

# Metrics by category
test_df = df_test.copy()
test_df['predicted_gap'] = y_pred


def get_metrics_by_category(df, gap_col, pred_col):
    metrics = {}
    for category in df['gap_category'].unique():
        mask = df['gap_category'] == category
        true_vals = df.loc[mask, gap_col]
        pred_vals = df.loc[mask, pred_col]

        if len(true_vals) > 0:
            mae = mean_absolute_error(true_vals, pred_vals)
            metrics[category] = {
                'mae': mae,
                'n_samples': len(true_vals),
                'true_mean': true_vals.mean(),
                'pred_mean': pred_vals.mean()
            }
    return metrics


category_metrics = get_metrics_by_category(test_df, gap_col, 'predicted_gap')

print(f"EXTERNAL TEST SET METRICS:")
print(f"   MAE:  {test_mae:.4f} eV")
print(f"   RMSE: {test_rmse:.4f} eV")
print(f"   R2:   {test_r2:.4f}")

print(f"\nMETRICS BY CATEGORY:")
for category, metrics in category_metrics.items():
    print(f"   {category:25}: MAE = {metrics['mae']:.4f} eV (n={metrics['n_samples']})")

# 7. PHYSICAL CONSISTENCY ANALYSIS
print("\nPHYSICAL CONSISTENCY ANALYSIS:")

# Verify there are no negative band gaps
negative_predictions = np.sum(y_pred < 0)
print(f"   Negative predictions: {negative_predictions} (should be 0)")

# Metallic materials analysis
metallic_mask = y_test == 0
if np.any(metallic_mask):
    metallic_mae = mean_absolute_error(y_test[metallic_mask], y_pred[metallic_mask])
    metallic_mean_pred = y_pred[metallic_mask].mean()
    print(f"   MAE for metallic materials: {metallic_mae:.4f} eV")
    print(f"   Average predicted band gap for metallics: {metallic_mean_pred:.4f} eV")

# 8. RESULTS VISUALIZATION
print("\nGENERATING VISUALIZATIONS...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Predictions vs Actual values
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=20)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
axes[0, 0].set_xlabel('Experimental Band Gap (eV)')
axes[0, 0].set_ylabel('Predicted Band Gap (eV)')
axes[0, 0].set_title('Predictions vs Actual Values')
axes[0, 0].grid(True, alpha=0.3)

# 2. Prediction error
error = y_pred - y_test
axes[0, 1].hist(error, bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('Prediction Error (eV)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Error Distribution')
axes[0, 1].grid(True, alpha=0.3)

# 3. MAE by category
categories = list(category_metrics.keys())
mae_values = [metrics['mae'] for metrics in category_metrics.values()]
axes[0, 2].bar(categories, mae_values, alpha=0.7)
axes[0, 2].set_xlabel('Category')
axes[0, 2].set_ylabel('MAE (eV)')
axes[0, 2].set_title('MAE by Category')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].grid(True, alpha=0.3)

# 4. Training evolution
axes[1, 0].plot(final_history.history['loss'], label='Training Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].set_title('Loss During Training')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5. Band gap distribution
axes[1, 1].hist(y_test, bins=50, alpha=0.7, label='Actual', edgecolor='black')
axes[1, 1].hist(y_pred, bins=50, alpha=0.7, label='Predicted', edgecolor='black')
axes[1, 1].set_xlabel('Band Gap (eV)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Band Gap Distribution')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 6. Cross-validation results
axes[1, 2].plot(range(1, 6), cv_scores, 'o-', linewidth=2, markersize=8)
axes[1, 2].axhline(np.mean(cv_scores), color='red', linestyle='--', label=f'Mean: {np.mean(cv_scores):.4f}')
axes[1, 2].set_xlabel('Fold')
axes[1, 2].set_ylabel('Validation MAE')
axes[1, 2].set_title('Cross-Validation Results')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. SAVE MODEL AND RESULTS
print("\nSAVING MODEL AND RESULTS...")

# Save model (modern Keras format)
final_model.save('/content/bandgap_nn_model.keras')
print("Model saved: /content/bandgap_nn_model.keras")

# Save scaler
import joblib

joblib.dump(scaler, '/content/scaler.pkl')
print("Scaler saved: /content/scaler.pkl")

# Save test results
results_df = pd.DataFrame({
    'composition': df_test[comp_col],
    'true_bandgap': y_test,
    'predicted_bandgap': y_pred,
    'gap_category': df_test['gap_category'],
    'error': y_pred - y_test
})
results_df.to_csv('/content/test_predictions.csv', index=False)
print("Test predictions saved: /content/test_predictions.csv")

# Save metrics (convert numpy to Python native)
metrics_summary = {
    'test_mae': float(test_mae),
    'test_rmse': float(test_rmse),
    'test_r2': float(test_r2),
    'cv_mae_mean': float(np.mean(cv_scores)),
    'cv_mae_std': float(np.std(cv_scores)),
    'best_hyperparameters': study.best_params,
    'category_metrics': {
        category: {
            'mae': float(metrics['mae']),
            'n_samples': int(metrics['n_samples']),
            'true_mean': float(metrics['true_mean']),
            'pred_mean': float(metrics['pred_mean'])
        }
        for category, metrics in category_metrics.items()
    }
}

import json

with open('/content/model_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print("Metrics saved: /content/model_metrics.json")
