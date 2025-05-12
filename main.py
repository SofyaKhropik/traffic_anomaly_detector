import time

import pandas as pd
import pyshark
from VAEpredict.VAE_process import predict_loss


def extract_info(packet):
    info = packet.transport_layer
    if 'tcp' in packet:
        info += f" {packet.tcp.srcport} > {packet.tcp.dstport}"
        if 'flags' in packet.tcp.field_names:
            info += f" [{packet.tcp.flags}]"
        if 'seq' in packet.tcp.field_names and 'ack' in packet.tcp.field_names:
            info += f" Seq={packet.tcp.seq} Ack={packet.tcp.ack}"
        if 'window_size' in packet.tcp.field_names:
            info += f" Win={packet.tcp.window_size}"
        if 'tcp.options.timestamp.tsval' in packet:
            info += f" TSval={packet.tcp.options.timestamp.tsval} TSecr={packet.tcp.options.timestamp.tsecr}"
    return info


def capture_wifi_traffic(interface='Wi-Fi', capture_duration=5):
    cap = pyshark.LiveCapture(interface=interface, display_filter='ip')

    traffic_data = []
    start_time = time.time()
    while time.time() - start_time < capture_duration:
        for packet in cap.sniff_continuously():
            row_data = {
                'No.': packet.number,
                'Time': packet.sniff_time,
                'Source': packet.ip.src,
                'Destination': packet.ip.dst,
                'Protocol': packet.transport_layer,
                'Length': packet.length,
                'Info': extract_info(packet)
            }
            traffic_data.append(row_data)

            if time.time() - start_time >= capture_duration:
                break

    df = pd.DataFrame(traffic_data, columns=['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length', 'Info'])
    return df


def main():
    data = capture_wifi_traffic()
    data_with_losses_unscaled_test, anomalies_value, normals_value = predict_loss(data)
    print(data_with_losses_unscaled_test)
    print("Anomalies:\n")
    print(anomalies_value)


if __name__ == '__main__':
    main()
