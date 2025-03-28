import scapy.all as scapy
import pandas as pd
import os
import time
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def capture_traffic(duration=60, output_dir="traffic_logs", file_prefix="traffic_capture"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{file_prefix}_{timestamp}.pcap")

    logger.info(f"Capturing network traffic for {duration} seconds...")
    packets = scapy.sniff(filter="ip", timeout=duration)
    scapy.wrpcap(output_file, packets)
    logger.info(f"✅ Traffic captured and saved to {output_file}")
    return output_file


def process_pcap(input_file, output_dir="traffic_logs", file_prefix="traffic_data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.info(f"Processing pcap file: {input_file}")
    packets = scapy.rdpcap(input_file)
    data = []

    for pkt in packets:
        if pkt.haslayer(scapy.IP):
            data.append([
                datetime.fromtimestamp(float(pkt.time)),
                pkt[scapy.IP].src,
                pkt[scapy.IP].dst,
                pkt[scapy.IP].proto,
                pkt[scapy.IP].ttl,
                len(pkt)
            ])

    df = pd.DataFrame(data, columns=["timestamp", "src_ip", "dst_ip", "protocol", "ttl", "length"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{file_prefix}_{timestamp}.csv")
    df.to_csv(output_file, sep=';', index=False)

    logger.info(f"✅ Processed data saved to {output_file}")
    return output_file


def extract_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    grouped = df.groupby(['src_ip', 'dst_ip'])

    features = []
    for (src_ip, dst_ip), group in grouped:
        total_packets = len(group)
        total_length = group['length'].sum()
        unique_protocols = group['protocol'].nunique()
        first_timestamp = group['timestamp'].min()
        last_timestamp = group['timestamp'].max()
        duration = (last_timestamp - first_timestamp).total_seconds() if pd.notnull(first_timestamp) and pd.notnull(
            last_timestamp) else 0

        features.append([src_ip, dst_ip, total_packets, total_length, unique_protocols, duration])

    feature_matrix = pd.DataFrame(features,
                                  columns=['src_ip', 'dst_ip', 'total_packets', 'total_length', 'unique_protocols',
                                           'duration'])
    feature_matrix['src_ip'] = feature_matrix['src_ip'].astype('category').cat.codes
    feature_matrix['dst_ip'] = feature_matrix['dst_ip'].astype('category').cat.codes

    return feature_matrix


def continuous_capture(interval=86400, output_dir="traffic_logs"):
    while True:
        pcap_file = capture_traffic(duration=interval, output_dir=output_dir)
        csv_file = process_pcap(pcap_file, output_dir=output_dir)
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super Network Traffic Monitor")
    parser.add_argument("--duration", type=int, default=86400, help="Duration for each capture session in seconds")
    parser.add_argument("--output_dir", type=str, default="traffic_logs", help="Directory to save captured logs")
    parser.add_argument("--continuous", action="store_true", help="Enable continuous capturing")

    args = parser.parse_args()

    if args.continuous:
        continuous_capture(interval=args.duration, output_dir=args.output_dir)
    else:
        pcap_file = capture_traffic(duration=args.duration, output_dir=args.output_dir)
        process_pcap(pcap_file, output_dir=args.output_dir)