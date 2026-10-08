"""Train design-conditioned shape-response meta-models.

Fits axis-wise networks from FEM response data across actuator designs.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from pathlib import Path

script_dir = Path(__file__).resolve().parent          # .../Shape-matching_opt/optimization
data_dir = script_dir/"data" # .../Shape-matching_opt/optimization/data
axis = "z"

# CSV file paths
data_path   = data_dir / f"restructured_sm_dataset_{axis}.csv"
design_path = data_dir / f"design_parameters_sm.csv"

# Output directories for models and scalers
model_dir  = script_dir / f"models_{axis}"

model_dir.mkdir(parents=True, exist_ok=True)


r2_log = []

# === Load datasets ===
dataset = pd.read_csv(data_path, header=None).values
design_params = pd.read_csv(design_path, header=None).values.T  # shape: (num_models, 3)
# remove the first column of design_params as it is fixed
design_params = design_params[:, 1:]  # shape: (num_models, 2)
num_models = dataset.shape[1] // 3


# === Train each small model ===
X_behaviors = []
Y_behaviors = []
for i in range(num_models):
    R, l = design_params[i]
    p = dataset[:, i * 3 + 0].reshape(-1, 1)
    u = dataset[:, i * 3 + 1].reshape(-1, 1)
    # stack X and Y behaviors by rows
    X_behavior = np.hstack([np.full_like(p, R), np.full_like(p, l), p, u])
    X_behaviors.append(X_behavior)
    print(X_behavior.shape)
    
    T = dataset[:, i * 3 + 2].reshape(-1, 1)
    Y_behaviors.append(T)



X_behaviors = np.vstack(X_behaviors)
Y_behaviors = np.vstack(Y_behaviors).reshape(-1, 1)
print(X_behaviors.shape, Y_behaviors.shape)
# === Normalize ===
x_behavior_scaler = StandardScaler().fit(X_behaviors)
y_behavior_scaler = StandardScaler().fit(Y_behaviors)

X_scaled = x_behavior_scaler.transform(X_behaviors)
Y_scaled = y_behavior_scaler.transform(Y_behaviors)

# === Train surrogate model ===
surrogate_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=0)
surrogate_model.fit(X_scaled, Y_scaled.ravel())

# === Evaluate ===
Y_pred_scaled = surrogate_model.predict(X_scaled)
Y_pred = y_behavior_scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1))
r2 = r2_score(Y_behaviors, Y_pred)
print(f"📈 Surrogate model R² score: {r2:.4f}")

# === Save scalers ===
np.save(os.path.join(model_dir, f"x_mean_{axis}.npy"), x_behavior_scaler.mean_)
np.save(os.path.join(model_dir, f"x_std_{axis}.npy"), x_behavior_scaler.scale_)
np.save(os.path.join(model_dir, f"y_mean_{axis}.npy"), y_behavior_scaler.mean_)
np.save(os.path.join(model_dir, f"y_std_{axis}.npy"), y_behavior_scaler.scale_)

# === Convert to TFLite ===
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

keras_model = Sequential()
keras_model.add(Dense(64, activation='relu', input_shape=(4,)))
keras_model.add(Dense(64, activation='relu'))
keras_model.add(Dense(1))
keras_model.compile(optimizer='adam', loss='mse')

keras_model.layers[0].set_weights([surrogate_model.coefs_[0], surrogate_model.intercepts_[0]])
keras_model.layers[1].set_weights([surrogate_model.coefs_[1], surrogate_model.intercepts_[1]])
keras_model.layers[2].set_weights([surrogate_model.coefs_[2], surrogate_model.intercepts_[2]])

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

with open(os.path.join(model_dir, f"behavior_model_{axis}.tflite"), "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model and scalers saved to the model_dir")