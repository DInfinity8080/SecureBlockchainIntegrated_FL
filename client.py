import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import flwr as fl
import tensorflow as tf
import numpy as np
import sys
import time
from model import create_model
from data_loader import load_and_preprocess, partition_data
from blockchain_helper import BlockchainHelper

# ── Device Tier Definitions ──────────────────────────────────────
DEVICE_TIERS = {
    1: {
        "name": "Powerful Edge Server",
        "epochs": 5,
        "batch_size": 16,
        "data_fraction": 1.0,
        "learning_rate": 0.001,
        "dropout_prob": 0.0,
    },
    2: {
        "name": "Mid-Range IoT Device",
        "epochs": 3,
        "batch_size": 32,
        "data_fraction": 0.7,
        "learning_rate": 0.001,
        "dropout_prob": 0.1,
    },
    3: {
        "name": "Constrained Sensor Node",
        "epochs": 1,
        "batch_size": 64,
        "data_fraction": 0.4,
        "learning_rate": 0.0005,
        "dropout_prob": 0.25,
    },
}

TIER_DISTRIBUTION = [1, 2, 2, 3, 3]


def assign_tier(client_id, num_clients=5, random_tiers=False):
    if random_tiers:
        rng = np.random.RandomState(client_id)
        return rng.choice([1, 2, 3], p=[0.2, 0.4, 0.4])
    return TIER_DISTRIBUTION[client_id % len(TIER_DISTRIBUTION)]


def compute_update_size(weights):
    total_bytes = 0
    for w in weights:
        total_bytes += w.nbytes
    return total_bytes


class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test,
                 blockchain=None, tier=2, simulate_dropout=True):
        self.client_id = client_id
        self.tier = tier
        self.tier_config = DEVICE_TIERS[tier]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = create_model(
            input_dim=x_train.shape[1],
            learning_rate=self.tier_config["learning_rate"]
        )
        self.blockchain = blockchain
        self.simulate_dropout = simulate_dropout
        self.rounds_participated = 0
        self.rounds_dropped = 0

        print(f"[Client {client_id}] Tier {tier}: {self.tier_config['name']}")
        print(f"  Epochs: {self.tier_config['epochs']}, "
              f"Batch: {self.tier_config['batch_size']}, "
              f"Data fraction: {self.tier_config['data_fraction']}, "
              f"LR: {self.tier_config['learning_rate']}")
        print(f"  Dropout probability: {self.tier_config['dropout_prob']}")
        print(f"  Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # ── Simulate Dropout (instant return, no training) ───────
        if self.simulate_dropout:
            dropout_prob = self.tier_config["dropout_prob"]
            if np.random.random() < dropout_prob:
                self.rounds_dropped += 1
                print(f"[Client {self.client_id}] DROPOUT — simulating "
                      f"connection failure (Tier {self.tier}, "
                      f"p={dropout_prob})")
                # Return stale weights with 0 examples = server knows this is a drop
                return self.model.get_weights(), 0, {"dropped": True}

        self.rounds_participated += 1
        self.model.set_weights(parameters)

        self.model.fit(
            self.x_train, self.y_train,
            epochs=self.tier_config["epochs"],
            batch_size=self.tier_config["batch_size"],
            verbose=1
        )

        updated_weights = self.model.get_weights()

        update_size_bytes = compute_update_size(updated_weights)
        update_size_kb = update_size_bytes / 1024
        print(f"[Client {self.client_id}] Update size: {update_size_kb:.1f} KB")

        if self.blockchain:
            try:
                _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                self.blockchain.submit_model_update(
                    updated_weights, accuracy,
                    account_index=self.client_id + 1
                )
                print(f"[Client {self.client_id}] Model update recorded on blockchain")
            except Exception as e:
                print(f"[Client {self.client_id}] Blockchain error: {e}")

        return updated_weights, len(self.x_train), {
            "tier": self.tier,
            "tier_name": self.tier_config["name"],
            "epochs": self.tier_config["epochs"],
            "batch_size": self.tier_config["batch_size"],
            "data_fraction": self.tier_config["data_fraction"],
            "update_size_bytes": update_size_bytes,
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"[Client {self.client_id}] (Tier {self.tier}) "
              f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.x_test), {"accuracy": accuracy, "tier": self.tier}


def start_client(client_id=0, num_clients=5, server_address="127.0.0.1:9090",
                 simulate_dropout=True, random_tiers=False):
    print(f"Starting Client {client_id}...")

    tier = assign_tier(client_id, num_clients, random_tiers=random_tiers)

    X, y = load_and_preprocess()
    partitions = partition_data(X, y, num_clients=num_clients)

    x_data, y_data = partitions[client_id]

    data_fraction = DEVICE_TIERS[tier]["data_fraction"]
    if data_fraction < 1.0:
        num_samples = int(len(x_data) * data_fraction)
        indices = np.random.choice(len(x_data), num_samples, replace=False)
        x_data = x_data[indices]
        y_data = y_data[indices]

    split = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    print(f"[Client {client_id}] Train: {len(x_train)}, Test: {len(x_test)}")

    try:
        blockchain = BlockchainHelper()
        device_id = f"client_{client_id}"
        blockchain.register_device(device_id, account_index=client_id + 1)
        print(f"[Client {client_id}] Registered on blockchain")
    except Exception as e:
        print(f"[Client {client_id}] Blockchain error: {e}")
        blockchain = None

    client = FLClient(client_id, x_train, y_train, x_test, y_test,
                      blockchain, tier=tier, simulate_dropout=simulate_dropout)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

    total_rounds = client.rounds_participated + client.rounds_dropped
    if total_rounds > 0:
        print(f"\n[Client {client_id}] Session Summary:")
        print(f"  Tier: {tier} ({DEVICE_TIERS[tier]['name']})")
        print(f"  Participated: {client.rounds_participated}/{total_rounds} rounds")
        print(f"  Dropped out:  {client.rounds_dropped}/{total_rounds} rounds")


if __name__ == '__main__':
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    simulate_dropout = '--no-dropout' not in sys.argv
    random_tiers = '--random-tiers' in sys.argv
    start_client(client_id=client_id, num_clients=num_clients,
                 simulate_dropout=simulate_dropout, random_tiers=random_tiers)