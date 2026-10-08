"""Train a tendon-driven joint torque surrogate from FEM data.

Exports a TFLite model and input/output scalers.
"""

import pandas as pd
import os

# Function to import TensorFlow while suppressing warnings
def import_tensorflow():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf

tf = import_tensorflow()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent

csv_path = THIS_DIR / "tendon_driven_FEM_data.csv"
x_scaler_path = THIS_DIR / "x_scaler.joblib"
y_scaler_path = THIS_DIR / "y_scaler.joblib"
tflite_path = THIS_DIR / "tendon_driven_surrogate.tflite"


df = pd.read_csv(csv_path)

# Extract input (tendon force, angle) and output (net torque)
X = df.iloc[:, :2].values  # First two columns: tendon force (p) and angle (u)
Y = df.iloc[:, 2].values   # Third column: net torque (T_net)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

# Scale the features (important for neural networks)
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()
Y_train = y_scaler.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_test = y_scaler.transform(Y_test.reshape(-1, 1)).flatten()


model = Sequential([
    Dense(16, input_dim=2, activation='relu'),  # Reduce neurons
    Dense(8, activation='relu'),  # Fewer hidden layers
    Dense(1, activation='linear')  # Output layer
])


# Define optimizer
custom_lr = 0.005  # Learning rate
optimizer = Adam(learning_rate=custom_lr)

# Compile the model
model.compile(optimizer=optimizer, loss='mse')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=400, batch_size=16, callbacks=[early_stopping])

# Save the trained model and scalers

dump(x_scaler, x_scaler_path)
dump(y_scaler, y_scaler_path)



# Make predictions
Y_pred = model.predict(X_test)
Y_pred = y_scaler.inverse_transform(Y_pred)
Y_test_unscaled = y_scaler.inverse_transform(Y_test.reshape(-1, 1))

# Compute R-squared score
r2 = r2_score(Y_test_unscaled, Y_pred)
print(f"R-squared score: {r2:.4f}")

# Plot learning curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.show()

# Scatter plot of actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(Y_test_unscaled, Y_pred, alpha=0.7, edgecolors='k', label='Predicted vs Actual')
plt.plot([min(Y_test_unscaled), max(Y_test_unscaled)], [min(Y_test_unscaled), max(Y_test_unscaled)], 'r--')  # Diagonal line
plt.xlabel('Actual Net Torque')
plt.ylabel('Predicted Net Torque')
plt.title('Actual vs. Predicted Net Torque')
plt.legend()
plt.grid(True)
plt.show()

import tensorflow.lite as tflite

converter = tflite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

interpreter = tflite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def fast_tflite_predict(p, u):
    x = np.array([[p, u]], dtype=np.float32)
    x_scaled = x_scaler.transform(x).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], x_scaled)
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])

    y = y_scaler.inverse_transform(y_scaled.reshape(-1, 1))
    return float(y[0, 0])
