import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import urllib.request

NSLKDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

ATTACK_MAP = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'mailbomb': 'dos',
    'apache2': 'dos', 'processtable': 'dos', 'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
    'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l',
    'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'worm': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r'
}

CLASS_MAP = {'normal': 0, 'dos': 1, 'probe': 2, 'r2l': 3, 'u2r': 4}

def download_nslkdd(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    train_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt'
    test_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
    
    train_path = os.path.join(data_dir, 'KDDTrain+.txt')
    test_path = os.path.join(data_dir, 'KDDTest+.txt')
    
    if not os.path.exists(train_path):
        print("Downloading NSL-KDD training set...")
        urllib.request.urlretrieve(train_url, train_path)
    if not os.path.exists(test_path):
        print("Downloading NSL-KDD test set...")
        urllib.request.urlretrieve(test_url, test_path)
    
    return train_path, test_path

def load_and_preprocess(data_dir='data'):
    train_path, test_path = download_nslkdd(data_dir)
    
    df_train = pd.read_csv(train_path, header=None, names=NSLKDD_COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=NSLKDD_COLUMNS)
    
    df_train.drop('difficulty', axis=1, inplace=True)
    df_test.drop('difficulty', axis=1, inplace=True)
    
    df_train['label'] = df_train['label'].map(ATTACK_MAP).map(CLASS_MAP)
    df_test['label'] = df_test['label'].map(ATTACK_MAP).map(CLASS_MAP)
    
    df_train.dropna(subset=['label'], inplace=True)
    df_test.dropna(subset=['label'], inplace=True)
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values.astype(np.int32)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def partition_data(X, y, num_clients=3):
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, num_clients)
    
    partitions = []
    for split in splits:
        partitions.append((X[split], y[split]))
    
    return partitions

if __name__ == '__main__':
    X, y = load_and_preprocess()
    partitions = partition_data(X, y, num_clients=3)
    
    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")
    for i, (px, py) in enumerate(partitions):
        print(f"Partition {i}: {len(px)} samples")
