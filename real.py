import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scapy.all import sniff, IP
from tensorflow import keras
from tensorflow.keras.metrics import MeanSquaredError  # Для метрики MSE
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from datetime import datetime


INTERFACE = "eth0"
LOG_FILE = "anomalies_log.csv"
MODEL_PATH = "autoencoder_model.h5"
ENCODER_PATH = "anomaly_threshold.npy"
SCALER_PATH = "scaler.pkl"
ANOMALY_THRESHOLD = None


print("🔄 Загрузка модели...")


autoencoder = keras.models.load_model(MODEL_PATH, custom_objects={'MeanSquaredError': MeanSquaredError})
ip_encoder = np.load(ENCODER_PATH, allow_pickle=True).item()
scaler = pd.read_pickle(SCALER_PATH)


def preprocess_packet(packet):
    """ Преобразует пакет в нужный формат и нормализует данные """
    if IP in packet:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        protocol = packet[IP].proto
        ttl = packet[IP].ttl
        length = len(packet)


        src_ip_encoded = ip_encoder.transform([[src_ip]])[0][0] if src_ip in ip_encoder.categories_[0] else -1
        dst_ip_encoded = ip_encoder.transform([[dst_ip]])[0][0] if dst_ip in ip_encoder.categories_[0] else -1


        scaled_features = scaler.transform([[ttl, length]])[0]
        ttl_scaled, length_scaled = scaled_features

        return np.array([[src_ip_encoded, dst_ip_encoded, protocol, ttl_scaled, length_scaled]]), timestamp, src_ip, dst_ip, protocol, ttl, length
    return None


def reconstruction_error(model, X):
    X_pred = model.predict(X)
    errors = np.mean(np.power(X - X_pred, 2), axis=1)
    return errors


def packet_handler(packet):
    global ANOMALY_THRESHOLD

    data = preprocess_packet(packet)
    if data is None:
        return

    X_input, timestamp, src_ip, dst_ip, protocol, ttl, length = data
    error = reconstruction_error(autoencoder, X_input)[0]

    if error > ANOMALY_THRESHOLD:
        print(f"🚨 Аномалия! Ошибка: {error:.5f}, Порог: {ANOMALY_THRESHOLD:.5f}")
        print(f"⏳ {timestamp} | {src_ip} → {dst_ip} | Протокол: {protocol} | TTL: {ttl} | Длина: {length}")

        # Записываем аномалию в файл
        with open(LOG_FILE, "a") as f:
            f.write(f"{timestamp},{src_ip},{dst_ip},{protocol},{ttl},{length},{error:.5f}\n")


if __name__ == "__main__":
    print("🚀 Запуск сетевого мониторинга...")и
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,src_ip,dst_ip,protocol,ttl,length,error\n")
    try:
        ANOMALY_THRESHOLD = float(input("Введите порог аномалии (или нажмите Enter для значения по умолчанию): "))
    except ValueError:
        print("🔄 Вычисление порога аномалий...")
        normal_data = np.random.normal(loc=0.01, scale=0.005, size=1000)
        ANOMALY_THRESHOLD = np.mean(normal_data) + 2 * np.std(normal_data)
        print(f"✅ Установленный порог аномалий: {ANOMALY_THRESHOLD:.5f}")

    sniff(prn=packet_handler, iface=INTERFACE, store=False)
