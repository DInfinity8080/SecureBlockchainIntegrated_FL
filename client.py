import flwr as fl
import tensorflow as tf
import numpy as np
import sys
from model import create_model
from data_loader import load_and_preprocess, partition_data
from blockchain_helper import BlockchainHelper

class FLClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test, blockchain=None):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = create_model(input_dim=x_train.shape[1])
        self.blockchain = blockchain

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(
            self.x_train, self.y_train,
            epochs=3,
            batch_size=32,
            verbose=1
        )

        updated_weights = self.model.get_weights()

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

        return updated_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"[Client {self.client_id}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.x_test), {"accuracy": accuracy}

def start_client(client_id=0, server_address="127.0.0.1:9090"):
    print(f"Starting Client {client_id}...")

    X, y = load_and_preprocess()
    partitions = partition_data(X, y, num_clients=3)

    x_data, y_data = partitions[client_id]
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

    client = FLClient(client_id, x_train, y_train, x_test, y_test, blockchain)

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == '__main__':
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(client_id=client_id)
