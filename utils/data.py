import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from config import CLASSIFICATION_TYPE, LABELS

def load_data(path, time_features_only=True):
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if time_features_only:
        features = [
            "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
            "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
            "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
            "Init Fwd Win Byts", "Init Bwd Win Byts",
            "Subflow Fwd Byts", "Subflow Bwd Byts",
            "Subflow Fwd Pkts", "Subflow Bwd Pkts",

            "Fwd PSH Flags", "Bwd PSH Flags",
            "Fwd URG Flags", "Bwd URG Flags",
            "Flow Byts/s", "Flow Pkts/s",
            "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
            "Fwd Pkt Len Std", "Bwd Pkt Len Std",
            "Fwd Pkt Len Max", "Bwd Pkt Len Max",
            "Fwd Pkt Len Min", "Bwd Pkt Len Min",
            "Down/Up Ratio",
            "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
            "Fwd Byts/b Avg", "Bwd Byts/b Avg",
            "Fwd Pkts/b Avg", "Bwd Pkts/b Avg",
            "Fwd Blk Rate Avg", "Bwd Blk Rate Avg",
        ]
        df = df[features + ['Label']]
        # print unique values in 'Label' column
        # print("Unique values in 'Label' column:", df['Label'].unique())
    # Encode labels
    # one to all mapping
    if CLASSIFICATION_TYPE == "one_to_all":
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
    else:
        df['Label'] = df['Label'].map(LABELS)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df.drop(columns=['Label']))
    y = df['Label'].values

    # convert df to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # print(type(X_tensor[0:5]))

    # y_tensor = y_tensor.squeeze()

    print(f"Loaded {X_tensor.shape[0]} samples with {X_tensor.shape[1]} features.")

    return X_tensor, y_tensor

def filter_attack_only(x, y):
    # Keep only the samples with attack labels
    attack_labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    mask = torch.isin(y, attack_labels)
    return x[mask], y[mask]

def reduce_benign_samples(x, y, reduction_rate):
    benign_label = 0
    benign_mask = (y == benign_label)
    benign_samples = x[benign_mask]
    benign_labels = y[benign_mask]
    # randomize
    perm = torch.randperm(benign_samples.size(0), device=benign_samples.device)
    benign_samples = benign_samples[perm]
    benign_labels = benign_labels[perm]
    keep_n = int(len(benign_samples) * reduction_rate)
    reduced_benign_samples = benign_samples[:keep_n]
    reduced_benign_labels = benign_labels[:keep_n]
    attack_mask = ~benign_mask
    return torch.cat((reduced_benign_samples, x[attack_mask])), torch.cat((reduced_benign_labels, y[attack_mask]))
