import pandas as pd
from VAEpredict.VAE_process import predict_loss


def custom_csv_parser(file_path):
    traffic_data = []
    with open(file_path, 'r') as file:
        header_processed = False
        for line in file:
            if not header_processed:
                header_processed = True
                continue

            comma_count = 0
            processed_line = []
            for char in line:
                if char == ',':
                    comma_count += 1
                if comma_count < 6:
                    processed_line.append(char)
                else:
                    break
            processed_data = ''.join(processed_line).strip().split(',')
            traffic_data.append(processed_data)

    df = pd.DataFrame(traffic_data, columns=['No.', 'Time', 'Source', 'Destination', 'Protocol', 'Length'])
    df['Length'] = df['Length'].astype(int)
    return df


def main():
    data = custom_csv_parser('traffic_ex.csv')
    data_with_losses_unscaled_test, anomalies_value, normals_value = predict_loss(data)
    print(data_with_losses_unscaled_test)
    print("Anomalies:\n")
    print(anomalies_value)


if __name__ == '__main__':
    main()