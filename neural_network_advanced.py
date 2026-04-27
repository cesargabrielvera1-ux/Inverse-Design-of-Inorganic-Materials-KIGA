"""
Neural Network Model 3 - Advanced Architecture
=================================================
Advanced neural network with extended optimization:
More layers, increased patience, wider search space, robust loss functions.

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
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

print("=" * 80)
print("ADVANCED NEURAL NETWORK - EXTENDED OPTIMIZATION")
print("IMPROVEMENTS: +Layers +Patience +SearchSpace +RobustLoss")
print("=" * 80)

# Reproducibility configuration
tf.random.set_seed(42)
np.random.seed(42)

# 1. LOAD AND PREPROCESS DATA (IMPROVED)
print("\nLOADING AND PREPROCESSING DATA (IMPROVED)...")

df_train = pd.read_csv('/content/train_balanced_stratified.csv')
df_test = pd.read_csv('/content/validation_balanced_stratified.csv')

print(f"Training set: {df_train.shape}")
print(f"Test set: {df_test.shape}")

comp_col = 'composition' if 'composition' in df_train.columns else 'final_composition'
gap_col = 'gap expt'

# Separate features and target
magpie_columns = [col for col in df_train.columns if 'MagpieData' in col]
X_train = df_train[magpie_columns].values
y_train = df_train[gap_col].values
X_test = df_test[magpie_columns].values
y_test = df_test[gap_col].values

print(f"Magpie features: {len(magpie_columns)}")

# 2. ADVANCED PREPROCESSING
print("\nAPPLYING ADVANCED PREPROCESSING...")

# Scaling for features
scaler_standard = StandardScaler()
X_train_scaled = scaler_standard.fit_transform(X_train)
X_test_scaled = scaler_standard.transform(X_test)

# Target transformation to improve distribution
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

print("Advanced preprocessing completed")

# 3. ADVANCED ARCHITECTURE WITH MORE OPTIONS
print("\nDEFINING ADVANCED ARCHITECTURE...")


def create_advanced_model(trial=None):
    """Creates an advanced model with an extended search space."""

    if trial is not None:
        # EXTENDED SEARCH SPACE
        n_layers = trial.suggest_int('n_layers', 3, 12)  # More layers
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.7)
        learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128, 256])

        # New optimizable parameters
        activation = trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'swish', 'leaky_relu'])
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'nadam', 'adamw', 'rmsprop'])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        use_residual = trial.suggest_categorical('use_residual', [True, False])
        l2_reg = trial.suggest_float('l2_reg', 1e-8, 1e-2, log=True)

        # Optimizable loss function
        loss_function = trial.suggest_categorical('loss_function', ['mae', 'mse', 'huber', 'log_cosh'])

        # More flexible dynamic architecture
        layer_sizes = []
        for i in range(n_layers):
            layer_sizes.append(trial.suggest_int(f'n_units_layer_{i}', 8, 1024))

    else:
        # Default values
        n_layers = 6
        dropout_rate = 0.3
        learning_rate = 1e-3
        batch_size = 32
        activation = 'swish'
        optimizer_name = 'adamw'
        use_batch_norm = True
        use_residual = False
        l2_reg = 1e-4
        loss_function = 'huber'
        layer_sizes = [512, 256, 128, 64, 32, 16]

    # Build model with flexible architecture
    inputs = layers.Input(shape=(X_train_scaled.shape[1],))
    x = inputs

    # Hidden layers with advanced options
    for i, units in enumerate(layer_sizes):
        # Dense layer with L2 regularization
        x_new = layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            kernel_initializer='he_normal'
        )(x)

        # Optional batch normalization
        if use_batch_norm:
            x_new = layers.BatchNormalization()(x_new)

        # Optional residual connections
        if use_residual and i > 0 and x.shape[-1] == units:
            x_new = layers.Add()([x, x_new])

        x = x_new
        x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(1, activation='softplus')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer selection
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_name == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_reg)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    # Loss function selection
    if loss_function == 'mae':
        loss = 'mae'
    elif loss_function == 'mse':
        loss = 'mse'
    elif loss_function == 'huber':
        loss = keras.losses.Huber(delta=1.0)
    elif loss_function == 'log_cosh':
        loss = keras.losses.LogCosh()

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['mae', 'mse']
    )

    return model, batch_size


# 4. IMPROVED LOSS FUNCTION
print("\nIMPLEMENTING IMPROVED LOSS FUNCTION...")


def weighted_mae_loss(y_true, y_pred):
    """Custom loss that gives more weight to errors in metallic materials."""
    # Additional weight for materials with zero band gap
    metallic_weight = tf.where(y_true == 0, 2.0, 1.0)
    # Additional weight for large errors
    error_weight = tf.where(tf.abs(y_true - y_pred) > 1.0, 1.5, 1.0)

    combined_weight = metallic_weight * error_weight
    return tf.reduce_mean(combined_weight * tf.abs(y_true - y_pred))


# 5. EXTENDED BAYESIAN OPTIMIZATION
print("\nCONFIGURING EXTENDED BAYESIAN OPTIMIZATION...")


def advanced_objective(trial):
    """Improved objective function for Optuna."""

    # Create model with suggested hyperparameters
    model, batch_size = create_advanced_model(trial)

    # Improved callbacks with greater patience
    callbacks = [
        TFKerasPruningCallback(trial, 'val_mae'),
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=50,  # Greater patience
            restore_best_weights=True,
            min_delta=0.0001,  # Higher sensitivity
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=25,  # Greater patience for LR reduction
            min_lr=1e-8,
            cooldown=5
        )
    ]

    # Training with more epochs and better monitoring
    history = model.fit(
        X_train_scaled, y_train,  # Use original target for loss
        batch_size=batch_size,
        epochs=500,  # More epochs
        validation_split=0.15,  # Slightly more validation
        callbacks=callbacks,
        verbose=0,
        shuffle=True
    )

    # Composite score that considers multiple metrics
    best_val_mae = min(history.history['val_mae'])
    best_val_loss = min(history.history['val_loss'])

    # Score that penalizes overfitting
    train_final_mae = history.history['mae'][-1]
    overfitting_penalty = max(0, (best_val_mae - train_final_mae) / train_final_mae)

    final_score = best_val_mae * (1 + 0.1 * overfitting_penalty)
    return final_score


# Run extended Bayesian optimization
print("STARTING EXTENDED BAYESIAN OPTIMIZATION (200 iterations)...")

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=500, reduction_factor=3
    ),
    sampler=optuna.samplers.TPESampler(seed=42)
)

# More extensive optimization
study.optimize(advanced_objective, n_trials=500, timeout=3600 * 10)  # 10 hours maximum

print("Extended Bayesian optimization completed")
print(f"Best hyperparameters: {study.best_params}")
print(f"Best validation MAE: {study.best_value:.4f}")

# 6. MORE RIGOROUS CROSS-VALIDATION
print("\nRUNNING MORE RIGOROUS CROSS-VALIDATION (10-FOLD)...")

# Configure K-Fold with more splits
kfold = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold for more robustness
cv_scores = []
cv_histories = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled)):
    print(f"Processing Fold {fold + 1}/10...")

    # Split data
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

    # Create model with best hyperparameters
    best_params = study.best_params
    model, batch_size = create_advanced_model(None)

    # Manually apply best hyperparameters
    if best_params:
        inputs = layers.Input(shape=(X_train_scaled.shape[1],))
        x = inputs

        for i in range(best_params['n_layers']):
            units = best_params[f'n_units_layer_{i}']

            x_new = layers.Dense(
                units,
                activation=best_params['activation'],
                kernel_regularizer=keras.regularizers.l2(best_params['l2_reg']),
                kernel_initializer='he_normal'
            )(x)

            if best_params['use_batch_norm']:
                x_new = layers.BatchNormalization()(x_new)

            if best_params['use_residual'] and i > 0 and x.shape[-1] == units:
                x_new = layers.Add()([x, x_new])

            x = x_new
            x = layers.Dropout(best_params['dropout_rate'])(x)

        outputs = layers.Dense(1, activation='softplus')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Optimizer
        if best_params['optimizer'] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        elif best_params['optimizer'] == 'nadam':
            optimizer = keras.optimizers.Nadam(learning_rate=best_params['learning_rate'])
        elif best_params['optimizer'] == 'adamw':
            optimizer = keras.optimizers.AdamW(learning_rate=best_params['learning_rate'],
                                             weight_decay=best_params['l2_reg'])
        elif best_params['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=best_params['learning_rate'])

        # Loss function
        if best_params['loss_function'] == 'mae':
            loss = 'mae'
        elif best_params['loss_function'] == 'mse':
            loss = 'mse'
        elif best_params['loss_function'] == 'huber':
            loss = keras.losses.Huber(delta=1.0)
        elif best_params['loss_function'] == 'log_cosh':
            loss = keras.losses.LogCosh()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
        batch_size = best_params['batch_size']

    # Callbacks with maximum patience
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=60,  # Maximum patience
            restore_best_weights=True,
            min_delta=0.0001,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.3,  # More aggressive reduction
            patience=30,
            min_lr=1e-9,
            cooldown=10
        )
    ]

    # Train with more epochs
    history = model.fit(
        X_fold_train, y_fold_train,
        batch_size=batch_size,
        epochs=500,  # More epochs
        validation_data=(X_fold_val, y_fold_val),
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate
    val_mae = model.evaluate(X_fold_val, y_fold_val, verbose=0)[1]
    cv_scores.append(val_mae)
    cv_histories.append(history)

    print(f"   Fold {fold + 1} - Validation MAE: {val_mae:.4f}")

print(f"\nCROSS-VALIDATION RESULTS (10-FOLD):")
print(f"   Average MAE: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
print(f"   Range: {np.min(cv_scores):.4f} - {np.max(cv_scores):.4f}")
print(f"   Coefficient of variation: {np.std(cv_scores)/np.mean(cv_scores)*100:.2f}%")

# 7. FINAL TRAINING WITH IMPROVED STRATEGY
print("\nTRAINING IMPROVED FINAL MODEL...")

# Create final model with best hyperparameters
best_params = study.best_params
final_model, final_batch_size = create_advanced_model(None)

# Apply best hyperparameters
if best_params:
    inputs = layers.Input(shape=(X_train_scaled.shape[1],))
    x = inputs

    for i in range(best_params['n_layers']):
        units = best_params[f'n_units_layer_{i}']

        x_new = layers.Dense(
            units,
            activation=best_params['activation'],
            kernel_regularizer=keras.regularizers.l2(best_params['l2_reg']),
            kernel_initializer='he_normal'
        )(x)

        if best_params['use_batch_norm']:
            x_new = layers.BatchNormalization()(x_new)

        if best_params['use_residual'] and i > 0 and x.shape[-1] == units:
            x_new = layers.Add()([x, x_new])

        x = x_new
        x = layers.Dropout(best_params['dropout_rate'])(x)

    outputs = layers.Dense(1, activation='softplus')(x)
    final_model = keras.Model(inputs=inputs, outputs=outputs)

    # Optimizer
    if best_params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    elif best_params['optimizer'] == 'nadam':
        optimizer = keras.optimizers.Nadam(learning_rate=best_params['learning_rate'])
    elif best_params['optimizer'] == 'adamw':
        optimizer = keras.optimizers.AdamW(learning_rate=best_params['learning_rate'],
                                         weight_decay=best_params['l2_reg'])
    elif best_params['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=best_params['learning_rate'])

    # Loss function
    if best_params['loss_function'] == 'mae':
        loss = 'mae'
    elif best_params['loss_function'] == 'mse':
        loss = 'mse'
    elif best_params['loss_function'] == 'huber':
        loss = keras.losses.Huber(delta=1.0)
    elif best_params['loss_function'] == 'log_cosh':
        loss = keras.losses.LogCosh()

    final_model.compile(optimizer=optimizer, loss=loss, metrics=['mae', 'mse'])
    final_batch_size = best_params['batch_size']

# Improved final callbacks
final_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='mae',
        patience=80,  # Extreme patience
        restore_best_weights=True,
        min_delta=0.0001,
        mode='min'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='mae',
        factor=0.2,  # Very aggressive reduction
        patience=40,
        min_lr=1e-10,
        cooldown=20
    ),
    keras.callbacks.ModelCheckpoint(
        '/content/best_model_advanced.keras',
        monitor='mae',
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    keras.callbacks.CSVLogger('/content/training_log_advanced.csv')
]

# Extended final training
final_history = final_model.fit(
    X_train_scaled, y_train,
    batch_size=final_batch_size,
    epochs=1500,  # Maximum epochs
    callbacks=final_callbacks,
    verbose=1,
    validation_split=0.1
)

print("Improved final training completed")

# Load the best saved model
final_model = keras.models.load_model('/content/best_model_advanced.keras')

# 8. ADVANCED EVALUATION ON TEST SET
print("\nADVANCED EVALUATION ON TEST SET...")

# Predict on test set
y_pred = final_model.predict(X_test_scaled).flatten()

# Main metrics
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

# Additional metrics
test_mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100
max_error = np.max(np.abs(y_test - y_pred))

print(f"ADVANCED TEST SET METRICS:")
print(f"   MAE:       {test_mae:.4f} eV")
print(f"   RMSE:      {test_rmse:.4f} eV")
print(f"   R2:        {test_r2:.4f}")
print(f"   MAPE:      {test_mape:.2f}%")
print(f"   Max Error: {max_error:.4f} eV")

# 9. IMPROVED DETAILED ANALYSIS BY CATEGORY
print("\nIMPROVED DETAILED ANALYSIS BY CATEGORY...")

test_df = df_test.copy()
test_df['predicted_gap'] = y_pred
test_df['abs_error'] = np.abs(y_pred - y_test)


def advanced_category_analysis(df, gap_col, pred_col):
    metrics = {}
    for category in df['gap_category'].unique():
        mask = df['gap_category'] == category
        true_vals = df.loc[mask, gap_col]
        pred_vals = df.loc[mask, pred_col]
        abs_errors = df.loc[mask, 'abs_error']

        if len(true_vals) > 0:
            metrics[category] = {
                'mae': mean_absolute_error(true_vals, pred_vals),
                'rmse': np.sqrt(mean_squared_error(true_vals, pred_vals)),
                'r2': r2_score(true_vals, pred_vals),
                'mape': np.mean(np.abs((true_vals - pred_vals) / np.where(true_vals != 0, true_vals, 1))) * 100,
                'n_samples': len(true_vals),
                'true_mean': true_vals.mean(),
                'pred_mean': pred_vals.mean(),
                'error_std': abs_errors.std(),
                'max_error': abs_errors.max(),
                'q95_error': np.quantile(abs_errors, 0.95)
            }
    return metrics


category_metrics = advanced_category_analysis(test_df, gap_col, 'predicted_gap')

print(f"\nDETAILED METRICS BY CATEGORY:")
for category, metrics in category_metrics.items():
    print(f"   {category:25}:")
    print(f"     MAE: {metrics['mae']:.4f} eV | RMSE: {metrics['rmse']:.4f} eV | R2: {metrics['r2']:.4f}")
    print(f"     MAPE: {metrics['mape']:.1f}% | Max Error: {metrics['max_error']:.4f} eV")
    print(f"     Samples: {metrics['n_samples']}")

# 10. ADVANCED PHYSICAL CONSISTENCY ANALYSIS
print("\nADVANCED PHYSICAL CONSISTENCY ANALYSIS:")

# Verify negative predictions
negative_predictions = np.sum(y_pred < 0)
print(f"   Negative predictions: {negative_predictions}/{len(y_pred)}")

# Improved metallic materials analysis
metallic_mask = y_test == 0
if np.any(metallic_mask):
    metallic_pred = y_pred[metallic_mask]
    metallic_stats = {
        'mean': metallic_pred.mean(),
        'std': metallic_pred.std(),
        'min': metallic_pred.min(),
        'max': metallic_pred.max(),
        'q95': np.quantile(metallic_pred, 0.95),
        'above_0_1': np.sum(metallic_pred > 0.1),
        'above_0_5': np.sum(metallic_pred > 0.5)
    }

    print(f"   Metallic materials (n={np.sum(metallic_mask)}):")
    print(f"     Average predicted band gap: {metallic_stats['mean']:.4f} +/- {metallic_stats['std']:.4f} eV")
    print(f"     Range: [{metallic_stats['min']:.4f}, {metallic_stats['max']:.4f}] eV")
    print(f"     95th percentile: {metallic_stats['q95']:.4f} eV")
    print(f"     > 0.1 eV: {metallic_stats['above_0_1']} | > 0.5 eV: {metallic_stats['above_0_5']}")

# Improved outlier analysis
error = np.abs(y_pred - y_test)
outlier_thresholds = [0.5, 1.0, 2.0]
print(f"\n   OUTLIER ANALYSIS:")
for threshold in outlier_thresholds:
    outliers = np.sum(error > threshold)
    print(f"     Error > {threshold} eV: {outliers}/{len(y_test)} ({outliers/len(y_test)*100:.1f}%)")

# 11. ADVANCED VISUALIZATIONS
print("\nGENERATING ADVANCED VISUALIZATIONS...")

fig, axes = plt.subplots(3, 2, figsize=(20, 18))

# 1. Predictions vs Actual with percentiles
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)

axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=15, c=error, cmap='viridis')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8, linewidth=2)
axes[0, 0].plot(y_test, slope*y_test + intercept, 'b-', alpha=0.8, linewidth=1)
axes[0, 0].set_xlabel('Experimental Band Gap (eV)')
axes[0, 0].set_ylabel('Predicted Band Gap (eV)')
axes[0, 0].set_title(f'Predictions vs Actual\nR2 = {r_value**2:.4f}, Slope = {slope:.4f}')
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0], label='Absolute Error (eV)')

# 2. Error distribution with percentiles
axes[0, 1].hist(error, bins=100, alpha=0.7, edgecolor='black', density=True)
axes[0, 1].axvline(np.median(error), color='red', linestyle='--', label=f'Median: {np.median(error):.3f} eV')
axes[0, 1].axvline(np.quantile(error, 0.95), color='orange', linestyle='--', label=f'95%: {np.quantile(error, 0.95):.3f} eV')
axes[0, 1].set_xlabel('Absolute Error (eV)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Absolute Error Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Metrics by category (grouped bar)
categories = list(category_metrics.keys())
mae_values = [metrics['mae'] for metrics in category_metrics.values()]
rmse_values = [metrics['rmse'] for metrics in category_metrics.values()]

x = np.arange(len(categories))
width = 0.35
axes[1, 0].bar(x - width/2, mae_values, width, label='MAE', alpha=0.7, color='skyblue')
axes[1, 0].bar(x + width/2, rmse_values, width, label='RMSE', alpha=0.7, color='lightcoral')
axes[1, 0].set_xlabel('Category')
axes[1, 0].set_ylabel('Error (eV)')
axes[1, 0].set_title('MAE and RMSE by Category')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories, rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Training evolution
axes[1, 1].plot(final_history.history['loss'], label='Training Loss', linewidth=2)
if 'val_loss' in final_history.history:
    axes[1, 1].plot(final_history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].set_title('Loss Evolution During Training')
axes[1, 1].set_yscale('log')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5. Band gap distribution
axes[2, 0].hist(y_test, bins=50, alpha=0.7, label='Actual', density=True, edgecolor='black')
axes[2, 0].hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True, edgecolor='black')
axes[2, 0].set_xlabel('Band Gap (eV)')
axes[2, 0].set_ylabel('Density')
axes[2, 0].set_title('Band Gap Distribution (Normalized)')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 6. Cross-validation results
axes[2, 1].plot(range(1, 11), cv_scores, 'o-', linewidth=2, markersize=6, color='green')
axes[2, 1].axhline(np.mean(cv_scores), color='red', linestyle='--',
                  label=f'Mean: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f} eV')
axes[2, 1].fill_between(range(1, 11),
                       np.mean(cv_scores) - np.std(cv_scores),
                       np.mean(cv_scores) + np.std(cv_scores),
                       alpha=0.2, color='red')
axes[2, 1].set_xlabel('Fold')
axes[2, 1].set_ylabel('Validation MAE (eV)')
axes[2, 1].set_title('Cross-Validation Results (10-Fold)')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/advanced_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. SAVE COMPLETE RESULTS
print("\nSAVING COMPLETE RESULTS...")

# Save model and components
final_model.save('/content/bandgap_nn_model_advanced.keras')
joblib.dump(scaler_standard, '/content/scaler_advanced.pkl')
joblib.dump(target_scaler, '/content/target_scaler_advanced.pkl')

# Save detailed results
advanced_results = {
    'test_metrics': {
        'mae': test_mae,
        'rmse': test_rmse,
        'r2': test_r2,
        'mape': test_mape,
        'max_error': max_error
    },
    'cv_metrics': {
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores),
        'scores': cv_scores,
        'cv_score': np.mean(cv_scores) + np.std(cv_scores)  # Conservative score
    },
    'category_metrics': category_metrics,
    'metallic_analysis': metallic_stats if np.any(metallic_mask) else {},
    'best_hyperparameters': study.best_params,
    'outliers_analysis': {
        f'error_above_{threshold}': np.sum(error > threshold) for threshold in outlier_thresholds
    },
    'regression_stats': {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'r_squared': r_value**2,
        'p_value': p_value
    }
}

with open('/content/advanced_results.json', 'w') as f:
    json.dump(advanced_results, f, indent=2, default=str)

# Save complete predictions
results_df = pd.DataFrame({
    'composition': df_test[comp_col],
    'true_bandgap': y_test,
    'predicted_bandgap': y_pred,
    'abs_error': error,
    'gap_category': df_test['gap_category'],
    'relative_error': np.where(y_test != 0, error / y_test, error)
})
results_df.to_csv('/content/complete_predictions_advanced.csv', index=False)

print("Advanced results saved:")
print("   - Model: /content/bandgap_nn_model_advanced.keras")
print("   - Scaler: /content/scaler_advanced.pkl")
print("   - Results: /content/advanced_results.json")
print("   - Predictions: /content/complete_predictions_advanced.csv")

# 13. IMPROVED FINAL SUMMARY
print("\n" + "=" * 80)
print("FINAL SUMMARY - ADVANCED OPTIMIZATION COMPLETED")
print("=" * 80)

print(f"MAIN IMPROVED METRICS:")
print(f"   - Test MAE:      {test_mae:.4f} eV")
print(f"   - Test RMSE:     {test_rmse:.4f} eV")
print(f"   - Test R2:       {test_r2:.4f}")
print(f"   - CV MAE:        {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f} eV")

print(f"\nEXPECTED IMPROVEMENT:")
improvement = 0.3329 - test_mae  # Compare with previous result
print(f"   - MAE reduction: {improvement:.4f} eV ({improvement/0.3329*100:.1f}%)")

print(f"\nOPTIMIZED ARCHITECTURE:")
print(f"   - Layers: {best_params['n_layers']}")
print(f"   - Neurons: {[best_params[f'n_units_layer_{i}'] for i in range(best_params['n_layers'])]}")
print(f"   - Activation: {best_params['activation']}")
print(f"   - Optimizer: {best_params['optimizer']}")

print(f"\nMODEL STABILITY:")
print(f"   - CV coefficient of variation: {np.std(cv_scores)/np.mean(cv_scores)*100:.2f}%")
print(f"   - CV range: {np.max(cv_scores) - np.min(cv_scores):.4f} eV")

print("\n" + "=" * 80)
