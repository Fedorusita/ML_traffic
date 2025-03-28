import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Функция подготовки данных
def preprocess_data(file_path, ip_encoder=None, scaler=None, fit=True):
    df = pd.read_csv(file_path, sep=';', encoding='utf-8').dropna()
    if ip_encoder is None:
        ip_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[['src_ip', 'dst_ip']] = ip_encoder.fit_transform(df[['src_ip', 'dst_ip']].astype(str))
    else:
        df[['src_ip', 'dst_ip']] = ip_encoder.transform(df[['src_ip', 'dst_ip']].astype(str))
    if scaler is None:
        scaler = MinMaxScaler()
        df[['ttl', 'length']] = scaler.fit_transform(df[['ttl', 'length']])
    else:
        df[['ttl', 'length']] = scaler.transform(df[['ttl', 'length']])

    return df[['src_ip', 'dst_ip', 'protocol', 'ttl', 'length']].values, ip_encoder, scaler



def create_autoencoder(input_dim):
    input_layer = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(16, activation='relu')(input_layer)
    encoded = keras.layers.Dense(8, activation='relu')(encoded)
    decoded = keras.layers.Dense(16, activation='relu')(encoded)
    output_layer = keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder



def train_autoencoder(X_train, epochs=50, batch_size=32):
    input_dim = X_train.shape[1]
    autoencoder = create_autoencoder(input_dim)

    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.1, verbose=1)

    return autoencoder



def reconstruction_error(model, X):
    X_pred = model.predict(X)
    errors = np.mean(np.power(X - X_pred, 2), axis=1)
    return errors



def calculate_anomaly_threshold(errors, std_multiplier=2):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    threshold = mean_error + std_multiplier * std_error
    return threshold



def detect_anomalies(model, data, threshold):
    errors = reconstruction_error(model, data)
    anomalies = errors > threshold
    return anomalies, errors



def plot_reconstruction_errors(errors, threshold):
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=50, alpha=0.7, color='blue', label="Ошибка восстановления")
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label=f'Порог: {threshold:.5f}')
    plt.xlabel("Ошибка восстановления")
    plt.ylabel("Количество пакетов")
    plt.title("Распределение ошибки восстановления")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_path = "traffic_logs/traffic_data_20250219_184335.csv"
    X, ip_enc, scaler = preprocess_data(file_path)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    autoencoder = train_autoencoder(X_train, epochs=50, batch_size=32)
    errors = reconstruction_error(autoencoder, X_test)
    threshold = calculate_anomaly_threshold(errors)
    print(f"✅ Установленный порог аномалии: {threshold:.5f}")
    plot_reconstruction_errors(errors, threshold)
    user_file = "test_user_data.csv"
    X_user, _, _ = preprocess_data(user_file, ip_enc, scaler, fit=False)
    anomalies, user_errors = detect_anomalies(autoencoder, X_user, threshold)
    print(f"🚨 Найдено {np.sum(anomalies)} аномалий в пользовательском датасете!")
