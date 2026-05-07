import pandas as pd
import numpy as np
import joblib
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class Spectrometer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.sample_data = None
        self.num_features = None  # store training feature count

    def initialize(self):
        print("CNN Spectrometer initialized.")

    # -----------------------------------------
    # Upload Sample Spectrum
    # -----------------------------------------
    def upload_sample_spectrum(self, file_path):
        print(f"Uploading sample spectrum from {file_path}...")

        df = pd.read_csv(file_path,header=None)
        df = df.select_dtypes(include=[np.number])

        self.sample_data = df.values
        print(f"Sample shape: {self.sample_data.shape}")

    # -----------------------------------------
    # Savitzky-Golay Safe Function
    # -----------------------------------------
    def apply_savgol(self, X):
        num_features = X.shape[1]

        if num_features < 3:
            raise ValueError("Too few features for smoothing")

        window_length = min(11, num_features)

        if window_length % 2 == 0:
            window_length -= 1

        if window_length <= 2:
            window_length = 3

        if window_length > num_features:
            window_length = num_features if num_features % 2 != 0 else num_features - 1

        return savgol_filter(X, window_length=window_length, polyorder=2, axis=1)

    # -----------------------------------------
    # Train Model
    # -----------------------------------------
    def train_model(self, file_path="only_s2.csv"):
        try:
            print("Loading dataset...")
            df = pd.read_csv(file_path)

            df = df.select_dtypes(include=[np.number])

            # Features and target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            self.num_features = X.shape[1]

            print(f"Training Data: X={X.shape}, y={y.shape}")

            # Smoothing
            X_smooth = self.apply_savgol(X)

            # Scaling
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_smooth)

            # Reshape for CNN
            X_cnn = X_scaled[..., np.newaxis]

            # Train/Test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_cnn, y, test_size=0.2, random_state=42
            )

            # CNN Model
            self.model = Sequential([
                Conv1D(32, 3, activation='relu',
                input_shape=(self.num_features, 1)),
                Conv1D(64, 3, activation='relu'),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dense(1)
            ])

            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )

            print("Training model...")
            self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=16,
                validation_data=(X_test, y_test),
                verbose=1
            )

            # Evaluation
            y_pred = self.model.predict(X_test).flatten()

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"RMSE: {rmse:.4f}")
            print(f"R²: {r2:.4f}")

            # Save model & scaler
            self.model.save("cnn_spectrometer_model.keras")
            joblib.dump(self.scaler, "cnn_scaler.pkl")
            joblib.dump(self.num_features, "cnn_features.pkl")

            print("Model saved successfully.")

        except Exception as e:
            print(f"Training error: {e}")

    # -----------------------------------------
    # Predict SoC
    # -----------------------------------------
    def predict_soc(self):
        try:
            # Load model if not loaded
            if self.model is None:
                print("Loading saved model...")
                self.model = load_model("cnn_spectrometer_model.keras")
                self.scaler = joblib.load("cnn_scaler.pkl")
                self.num_features = joblib.load("cnn_features.pkl")

            if self.sample_data is None:
                print("No sample data uploaded.")
                return None

            # 🔥 Feature consistency check
            if self.sample_data.shape[1] != self.num_features:
                print(f"Feature mismatch! Expected {self.num_features}, got {self.sample_data.shape[1]}")
                return None

            # Smoothing
            X_smooth = self.apply_savgol(self.sample_data)

            # Scaling
            X_scaled = self.scaler.transform(X_smooth)

            # Reshape
            X_cnn = X_scaled[..., np.newaxis]

            # Prediction
            prediction = self.model.predict(X_cnn)
            soc_value = float(prediction[0])

            print(f"Predicted SoC: {soc_value:.4f}")
            return soc_value

        except Exception as e:
            print(f"Prediction error: {e}")
            return None


# -----------------------------------------
# Example Usage
# -----------------------------------------
if __name__ == "__main__":
    spec = Spectrometer()
    spec.initialize()

    # Train model
    spec.train_model("only_s2.csv")

    # Predict
    spec.upload_sample_spectrum("sample.csv")
    spec.predict_soc()