import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split



def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def build_autoencoder(input_dim):
    encoder = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu")  # Сжатое представление
    ])

    decoder = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(16,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid")  # Восстановленный выход
    ])

    input_layer = keras.Input(shape=(input_dim,))
    encoded = encoder(input_layer)
    decoded = decoder(encoded)

    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")  # MSE измеряет ошибку восстановления

    return autoencoder



def train_autoencoder(data_path, model_path="autoencoder_model.h5", epochs=50, batch_size=32):
    df = load_data(data_path)

    X = df.to_numpy()

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)


    autoencoder = build_autoencoder(input_dim=X.shape[1])


    autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))


    autoencoder.save(model_path)
    print(f"Модель сохранена в {model_path}")



if __name__ == "__main__":
    train_autoencoder("processed_traffic_data.csv")
