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

_NSLKDD_URLS = {
    'KDDTrain+.txt': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
    'KDDTest+.txt':  'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt',
}

def download_nslkdd(data_dir='data'):
    os.makedirs(data_dir, exist_ok=True)
    paths = {}
    for filename, url in _NSLKDD_URLS.items():
        dest = os.path.join(data_dir, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename} ...")
            try:
                urllib.request.urlretrieve(url, dest)
                # Sanity-check: reject suspiciously small files (e.g. 404 HTML)
                if os.path.getsize(dest) < 10_000:
                    os.remove(dest)
                    raise RuntimeError(f"Downloaded file is too small — the URL may have changed.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to download NSL-KDD dataset ({filename}).\n"
                    f"  URL tried : {url}\n"
                    f"  Error     : {exc}\n"
                    f"  Fix       : Place {filename} manually in the '{data_dir}/' directory.\n"
                    f"  Source    : https://github.com/defcom17/NSL_KDD"
                ) from exc
        paths[filename] = dest
    return paths['KDDTrain+.txt'], paths['KDDTest+.txt']

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
    """IID partition: random uniform split across clients."""
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, num_clients)
    return [(X[s], y[s]) for s in splits]


def dirichlet_partition_data(X, y, num_clients=3, alpha=0.5, min_samples=10, seed=42):
    """Non-IID partition using a Dirichlet distribution over class labels.

    Lower *alpha* → more heterogeneous (each client dominated by fewer classes).
    Higher *alpha* → approaches IID.  Typical research values: 0.1 (very skewed),
    0.5 (moderate), 1.0 (mild).

    Parameters
    ----------
    X, y        : full dataset arrays
    num_clients : number of FL clients
    alpha       : Dirichlet concentration parameter (>0)
    min_samples : minimum samples guaranteed per client (via top-up from remainder)
    seed        : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    # Build per-class index lists
    class_indices = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx = class_indices[c]
        # Draw proportions from Dirichlet
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        # Convert to cumulative counts; clip negatives from rounding
        counts = np.maximum(0, (proportions * len(idx)).astype(int))
        # Fix rounding so counts sum exactly to len(idx)
        counts[-1] = max(0, len(idx) - counts[:-1].sum())
        split_points = np.cumsum(counts[:-1])
        splits = np.split(idx, split_points)
        for cid, chunk in enumerate(splits):
            client_indices[cid].extend(chunk.tolist())

    # Guarantee min_samples per client by redistributing from the richest client.
    # (There is no leftover remainder — Dirichlet distributes every index.)
    for cid in range(num_clients):
        deficit = min_samples - len(client_indices[cid])
        if deficit <= 0:
            continue
        # Find the client with the most surplus samples
        richest = max(
            (j for j in range(num_clients) if j != cid),
            key=lambda j: len(client_indices[j]),
        )
        available = len(client_indices[richest]) - min_samples
        move = min(deficit, max(0, available))
        if move > 0:
            take = client_indices[richest][:move]
            client_indices[richest] = client_indices[richest][move:]
            client_indices[cid].extend(take)

    partitions = []
    for cid in range(num_clients):
        # Use explicit dtype=int — np.array([]) defaults to float64
        idx = np.array(client_indices[cid], dtype=int)
        if len(idx) > 0:
            rng.shuffle(idx)
        partitions.append((X[idx], y[idx]))

    return partitions


if __name__ == '__main__':
    X, y = load_and_preprocess()
    print(f"Total samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")

    print("\n-- IID partitions --")
    for i, (px, py) in enumerate(partition_data(X, y, num_clients=3)):
        dist = {int(c): int((py == c).sum()) for c in np.unique(y)}
        print(f"  Client {i}: {len(px)} samples | class dist: {dist}")

    print("\n-- Non-IID Dirichlet partitions (alpha=0.5) --")
    for i, (px, py) in enumerate(dirichlet_partition_data(X, y, num_clients=3, alpha=0.5)):
        dist = {int(c): int((py == c).sum()) for c in np.unique(y)}
        print(f"  Client {i}: {len(px)} samples | class dist: {dist}")
