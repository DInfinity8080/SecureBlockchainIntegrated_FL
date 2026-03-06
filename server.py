import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from model import create_model
from blockchain_helper import BlockchainHelper
from poisoning_detector import PoisoningDetector

class SecureFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, blockchain, poisoning_detector, **kwargs):
        super().__init__(**kwargs)
        self.blockchain = blockchain
        self.detector = poisoning_detector
        self.global_weights = None
        self.client_account_map = {}
        self.devices_registered = False

    def _register_devices(self, num_clients):
        if self.devices_registered:
            return
        for i in range(num_clients):
            account_idx = i + 1
            self.client_account_map[i] = account_idx
            self.blockchain.ensure_registered(account_idx, f"fl_client_{i}")
        self.devices_registered = True
        print(f"Registered {num_clients} devices on blockchain")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Register devices on first round
        try:
            self._register_devices(len(results))
        except Exception as e:
            print(f"Device registration error: {e}")

        # Extract client updates
        client_updates = []
        client_info = []
        for i, (client_proxy, fit_res) in enumerate(results):
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append((i, weights))
            client_info.append((client_proxy, fit_res, i))

        # Run poisoning detection
        print(f"\n{'='*60}")
        print(f"[Round {server_round}] Running poisoning detection...")
        detection_results = self.detector.detect_poisoning(
            client_updates, self.global_weights
        )

        clean_ids = self.detector.get_clean_clients(detection_results)
        poisoned_ids = self.detector.get_poisoned_clients(detection_results)

        print(f"Clean clients: {clean_ids}")
        if poisoned_ids:
            print(f"POISONED clients detected: {poisoned_ids}")

        # Validate on blockchain and update reputations
        for client_id, info in detection_results.items():
            try:
                account_idx = self.client_account_map[client_id]
                device_addr = self.blockchain.accounts[account_idx]
                self.blockchain.validate_update(
                    device_addr, server_round - 1, info['passed_validation'],
                    owner_index=0
                )
                rep = self.blockchain.get_reputation(account_index=account_idx)
                print(f"[Client {client_id}] Reputation: {rep}")
            except Exception as e:
                print(f"[Client {client_id}] Blockchain validation error: {e}")

        # Get reputation weights for clean clients
        rep_weights = {}
        total_rep = 0
        for client_id in clean_ids:
            try:
                account_idx = self.client_account_map[client_id]
                rep = self.blockchain.get_reputation(account_index=account_idx)
                rep_weights[client_id] = rep
                total_rep += rep
            except:
                rep_weights[client_id] = 100
                total_rep += 100

        # Filter to clean clients only
        filtered_results = []
        for client_proxy, fit_res, client_id in client_info:
            if client_id in clean_ids:
                filtered_results.append((client_proxy, fit_res, client_id))

        if not filtered_results:
            print("WARNING: All clients flagged! Using all results.")
            filtered_results = client_info

        # Reputation-weighted averaging
        weights_results = []
        for client_proxy, fit_res, cid in filtered_results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            rep = rep_weights.get(cid, 100)
            weight_factor = rep / total_rep if total_rep > 0 else 1.0 / len(filtered_results)
            weights_results.append((client_weights, weight_factor))

        aggregated = [
            np.zeros_like(weights_results[0][0][i])
            for i in range(len(weights_results[0][0]))
        ]
        for client_weights, weight_factor in weights_results:
            for i, w in enumerate(client_weights):
                aggregated[i] += w * weight_factor

        self.global_weights = aggregated

        # Record on blockchain
        try:
            self.blockchain.update_global_model(aggregated, owner_index=0)
            print(f"[Round {server_round}] Global model recorded on blockchain")
        except Exception as e:
            print(f"[Round {server_round}] Blockchain error: {e}")

        print(f"{'='*60}\n")

        return ndarrays_to_parameters(aggregated), {}

def start_server(num_rounds=5):
    model = create_model()
    initial_weights = model.get_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    try:
        blockchain = BlockchainHelper()
        print("Blockchain connected for FL server")
    except Exception as e:
        print(f"Blockchain connection failed: {e}")
        blockchain = None

    detector = PoisoningDetector(z_threshold=1.5)

    if blockchain:
        strategy = SecureFedAvg(
            blockchain=blockchain,
            poisoning_detector=detector,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            initial_parameters=initial_parameters,
        )

    fl.server.start_server(
        server_address="0.0.0.0:9090",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == '__main__':
    start_server(num_rounds=5)
