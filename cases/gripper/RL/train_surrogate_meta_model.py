"""Train the gripper's design-conditioned surrogate meta-model.

Reads the precomputed FEM-derived coefficient dataset, trains a neural
network and exports the model and feature/target scalers for task simulation."""

import pandas as pd

import os
# replace `import tensorflow as tf` with this line
# or insert this line at the beginning of the `__init__.py` of a package that depends on tensorflow
def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore


from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.regularizers import l1, l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

script_directory = os.path.dirname(os.path.abspath(__file__))


# Load the dataset for torque_net, in which the first four cols are design parameters, the rest cols are:
# 5: m_act, 6: b_act, 7: m_ela, 8: b_ela, 9: k_torque_theta0

file_name = "poly_surrogate_coefficient.csv"
file_path = os.path.join(script_directory, file_name)


df_stiffness = pd.read_csv(file_path)


# Split df_stiffness into input (X) and output (Y)
X = df_stiffness.iloc[:, :4].values
Y = df_stiffness.iloc[:, 4:9].values

# only keep the data when X[i, 1] == 1.5
X2 = []
Y2 = []
for i in range(X.shape[0]):
    if X[i, 1] == 1.5:
        X2.append(X[i])
        Y2.append(Y[i])
X2 = np.array(X2)
Y2 = np.array(Y2)
# get rid of the second column in X2
X2 = np.delete(X2, 1, axis=1)
X = X2
print(" X after reshaping:", X)
print(" X shape after reshaping:", X.shape)
Y = Y2

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

# Scale the features (important for neural networks)
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

y_scaler = StandardScaler()
Y_train = y_scaler.fit_transform(Y_train)
Y_test = y_scaler.transform(Y_test)


# Create a Sequential model
model = Sequential()

# Add layers
model.add(Dense(64, input_dim=3, activation='relu'))  # Input layer with 3 neurons, and 'relu' activation
# Hidden layers
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(64, activation='relu'))  

model.add(Dense(5, activation='linear'))  # Output layer with 5 neurons (for 5 output parameters)
    

# Define the optimizer with a custom learning rate
custom_lr = 0.005  # Example learning rate
optimizer = Adam(learning_rate=custom_lr)

# Compile the model with the custom optimizer
model.compile(optimizer='adam', loss='mse')

# Define the early stopping callback
early_stopping = EarlyStopping(start_from_epoch = 100, monitor='val_loss', patience = 20, restore_best_weights=True)

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=400, batch_size=16, callbacks=[early_stopping])

# Evaluate the model

# Predicting
example_predictions = y_scaler.inverse_transform(model.predict(x_scaler.transform(np.array([[2.75, 6.75, 11]]))))
print('Example (2.75, 1.5, 6.75, 11) Predictions', example_predictions)

predictions = model.predict(X_test)
predictions = y_scaler.inverse_transform(predictions)
Y_test_unscaled = y_scaler.inverse_transform(Y_test)

Y_predictions = y_scaler.inverse_transform(model.predict(x_scaler.transform(X)))


# Prepare data for the learning curve
loss_data = {
    'Epoch': list(range(1, len(history.history['loss']) + 1)),
    'Training Loss': history.history['loss'],
    'Validation Loss': history.history['val_loss']
}

# Prepare data for scatter plot (actual vs predicted values)
scatter_all_data = {
    'Actual': Y.flatten(),
    'Prediction': Y_predictions.flatten()
}

scatter_test_data = {
    'Actual': Y_test_unscaled.flatten(),
    'Prediction': predictions.flatten()
}


from sklearn.metrics import r2_score
r2 = r2_score(Y_test_unscaled, predictions)
print("R-squared:", r2)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(Y_test_unscaled, predictions, alpha=0.7, label='Predicted vs Actual', edgecolors='k')
plt.scatter(Y, Y_predictions, alpha=0.7)
plt.plot([min(Y.flatten()), max(Y.flatten())],
         [min(Y.flatten()), max(Y.flatten())],
        )

# Add R-squared value to the plot

# Labels and legend
plt.xlabel('Actual')
plt.ylabel('Prediction')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Create DataFrames
loss_df = pd.DataFrame(loss_data)
scatter_all_df = pd.DataFrame(scatter_all_data)
scatter_test_df = pd.DataFrame(scatter_test_data)

# Save the DataFrames to CSV files
loss_path = os.path.join(script_directory, 'Learning_Curve_Data.csv')
loss_df.to_csv(loss_path, index=False)
scatter_all_path = os.path.join(script_directory, 'Actual_vs_Predictions_All_Data.csv')
scatter_all_df.to_csv(scatter_all_path, index=False)
scatter_test_path = os.path.join(script_directory, 'Actual_vs_Predictions_Test_Data.csv')
scatter_test_df.to_csv(scatter_test_path, index=False)

# Save the model and scaler
model_path = os.path.join(script_directory, 'poly_meta-model.h5')
model.save(model_path)
scale_path = os.path.join(script_directory, 'poly_meta-model_x_scaler.joblib')
dump(x_scaler, scale_path)
scale_path = os.path.join(script_directory, 'poly_meta-model_y_scaler.joblib')
dump(y_scaler, scale_path)
