"""Train surrogate models for selected optimized actuator designs.

Exports design-specific neural-network models and feature scalers.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from pathlib import Path

# === Paths ===
THIS_DIR = Path(__file__).resolve().parent          # .../Helical/Shape-matching_opt/result
RESULT_DIR = THIS_DIR      
DATA_DIR   = RESULT_DIR / "data"  # .../Helical/Shape-matching_opt/result/data                   
MODEL_DIR  = RESULT_DIR / "opt_models"
SCALER_DIR = RESULT_DIR / "opt_scalers"

axis = "z"

data_path   = DATA_DIR / f"restructured_sm_res_{axis}.csv"
design_path = DATA_DIR / "design_parameters_sm_res.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR.mkdir(parents=True, exist_ok=True)

r2_log = []

# === Load datasets ===
dataset = pd.read_csv(data_path, header=None).values
design_params = pd.read_csv(design_path, header=None).values.T

# remove the first column of design_params as it is fixed
design_params = design_params[:, 1:]  # shape: (num_models, 2)

# === Collect all input/output data across all designs ===
all_X, all_Y = [], []
num_models = dataset.shape[1] // 3
for i in range(num_models):
    p = dataset[:, i * 3 + 0].reshape(-1, 1)
    u = dataset[:, i * 3 + 1].reshape(-1, 1)
    T = dataset[:, i * 3 + 2].reshape(-1, 1)

    valid = ~np.isnan(p).flatten() & ~np.isnan(u).flatten() & ~np.isnan(T).flatten()
    X = np.hstack([p[valid], u[valid]])
    Y = T[valid]
    all_X.append(X)
    all_Y.append(Y)

all_X = np.vstack(all_X)
all_Y = np.vstack(all_Y)

# === Fit shared scalers ===
x_scaler = StandardScaler().fit(all_X)
y_scaler = StandardScaler().fit(all_Y)

x_mean, x_std = x_scaler.mean_, x_scaler.scale_
y_mean, y_std = y_scaler.mean_, y_scaler.scale_

np.save(SCALER_DIR / f"x_mean_{axis}.npy", x_mean)
np.save(SCALER_DIR / f"x_std_{axis}.npy",  x_std)
np.save(SCALER_DIR / f"y_mean_{axis}.npy", y_mean)
np.save(SCALER_DIR / f"y_std_{axis}.npy",  y_std)


# === Train each small model ===
for i in range(num_models):
    p = dataset[:, i * 3 + 0].reshape(-1, 1)
    u = dataset[:, i * 3 + 1].reshape(-1, 1)
    T = dataset[:, i * 3 + 2].reshape(-1, 1)

    valid = ~np.isnan(p).flatten() & ~np.isnan(u).flatten() & ~np.isnan(T).flatten()
    X = np.hstack([p[valid], u[valid]])
    Y = T[valid]

    X_scaled = x_scaler.transform(X)
    Y_scaled = y_scaler.transform(Y)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.005), loss='mse')
    model.fit(X_scaled, Y_scaled, epochs=200, verbose=0)

    Y_pred_scaled = model.predict(X_scaled)
    Y_pred = y_scaler.inverse_transform(Y_pred_scaled)
    r2 = r2_score(Y, Y_pred)

    R, l = design_params[i]
    tag = f"R{R:.1f}_l{l:.1f}"
    r2_log.append([tag, r2])

    # Export to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out_path = MODEL_DIR / f"small_model_{axis}_{tag}.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)



    print(f"✅ Trained and saved model, weights for design {tag}, R² = {r2:.4f}")

average_r2 = np.mean([r2 for _, r2 in r2_log])
print(f"📊 Average R² across all small models: {average_r2:.4f}")
