import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.signal import savgol_filter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_excel("only_s2.xlsx")

# Extract features and target variable
X = df.iloc[:, :-1].values  # Spectral data (all columns except last one)
y = df.iloc[:, -1].values   # Target variable (Soil Organic Carbon - SoC)

# Apply Savitzky-Golay filter (optional)
X_smooth = savgol_filter(X, window_length=11, polyorder=2, deriv=0, axis=1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_smooth)

# Reshape for CNN (Conv1D expects 3D input: [samples, timesteps, features])
X_cnn = X_scaled[..., np.newaxis]  # Shape becomes (samples, timesteps, 1)

# Split into training and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output layer
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the CNN model
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Predict on test set
y_pred = model.predict(X_test).flatten()

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = np.mean(np.abs(y_test - y_pred))

print(f"Test RMSE: {rmse:.4f}")
print(f"Test R² Score: {r2:.4f}")
print(f"Test MAE: {mae:.4f}")

# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolors='k', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r', lw=2)
plt.xlabel("Actual SoC Values")
plt.ylabel("Predicted SoC Values")
plt.title("CNN: Actual vs. Predicted Soil Organic Carbon")
plt.grid()
plt.show()

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("CNN Training Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()
