import flwr as fl
import tensorflow as tf
import numpy as np
import sys
from model import create_model
from data_loader import load_and_preprocess, partition_data
from blockchain_helper import BlockchainHelper

class MaliciousClient(fl.client.NumPyClient):
    def __init__(self, client_id, x_train, y_train, x_test, y_test, 
                 blockchain=None, attack_type='label_flip'):
        self.client_id = client_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.model = create_model(input_dim=x_train.shape[1])
        self.blockchain = blockchain
        self.attack_type = attack_type

        # Poison the training data
        if attack_type == 'label_flip':
            print(f"[ATTACKER {client_id}] Label flip attack active")
            self.y_train = (self.y_train + 1) % 5

        elif attack_type == 'noise_injection':
            print(f"[ATTACKER {client_id}] Noise injection attack active")
            self.x_train = self.x_train + np.random.normal(0, 2.0, self.x_train.shape)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(
            self.x_train, self.y_train,
            epochs=3, batch_size=32, verbose=1
        )

        updated_weights = self.model.get_weights()

        # Scale attack: amplify the update
        if self.attack_type == 'scaling':
            print(f"[ATTACKER {client_id}] Scaling attack - amplifying weights 10x")
            diff = [new - old for new, old in zip(updated_weights, parameters)]
            updated_weights = [old + 10.0 * d for old, d in zip(parameters, diff)]

        if self.blockchain:
            try:
                _, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                self.blockchain.submit_model_update(
                    updated_weights, accuracy,
                    account_index=self.client_id + 1
                )
            except Exception as e:
                print(f"[ATTACKER {client_id}] Blockchain error: {e}")

        return updated_weights, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"[ATTACKER {self.client_id}] Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.x_test), {"accuracy": accuracy}

def start_malicious_client(client_id=0, attack_type='label_flip', 
                            server_address="127.0.0.1:9090"):
    print(f"Starting MALICIOUS Client {client_id} ({attack_type})...")

    X, y = load_and_preprocess()
    partitions = partition_data(X, y, num_clients=5)

    x_data, y_data = partitions[client_id]
    split = int(0.8 * len(x_data))
    x_train, x_test = x_data[:split], x_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    try:
        blockchain = BlockchainHelper()
        device_id = f"malicious_client_{client_id}"
        blockchain.register_device(device_id, account_index=client_id + 1)
    except Exception as e:
        print(f"Blockchain error: {e}")
        blockchain = None

    client = MaliciousClient(
        client_id, x_train, y_train, x_test, y_test,
        blockchain, attack_type
    )

    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )

if __name__ == '__main__':
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    attack = sys.argv[2] if len(sys.argv) > 2 else 'label_flip'
    start_malicious_client(client_id=client_id, attack_type=attack)
