import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import numpy as np
import csv
import time
from typing import List, Tuple, Dict, Optional, Union
from model import create_model
from blockchain_helper import BlockchainHelper
from poisoning_detector import PoisoningDetector

# ── Configuration ────────────────────────────────────────────────
NUM_CLIENTS = 10
NUM_ROUNDS = 5
Z_THRESHOLD = 1.5
SERVER_ADDRESS = "0.0.0.0:9090"
COMM_METRICS_FILE = "communication_metrics.csv"
# ─────────────────────────────────────────────────────────────────


def compute_weights_size(weights):
    """Calculate the size of model weights in bytes."""
    total = 0
    for w in weights:
        total += w.nbytes
    return total


class SecureFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, blockchain, poisoning_detector, **kwargs):
        super().__init__(**kwargs)
        self.blockchain = blockchain
        self.detector = poisoning_detector
        self.global_weights = None
        self.client_account_map = {}
        self.devices_registered = False
        self.comm_log = []  # Track communication metrics per round

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
        round_comm_bytes = 0

        for i, (client_proxy, fit_res) in enumerate(results):
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append((i, weights))
            client_info.append((client_proxy, fit_res, i))

            # ── Communication Metrics ────────────────────────────
            update_size = compute_weights_size(weights)
            round_comm_bytes += update_size

        num_clients = len(results)
        # Global model broadcast size (sent to each client)
        global_model_size = compute_weights_size(
            parameters_to_ndarrays(results[0][1].parameters)
        )
        broadcast_bytes = global_model_size * num_clients
        total_round_bytes = round_comm_bytes + broadcast_bytes

        # Centralized equivalent: all raw data sent to server
        # NSL-KDD: ~148K samples × 41 features × 4 bytes = ~24.3 MB
        centralized_data_bytes = 148517 * 41 * 4

        print(f"\n[Round {server_round}] Communication Metrics:")
        print(f"  Client uploads:    {num_clients} × {global_model_size/1024:.1f} KB = {round_comm_bytes/1024:.1f} KB total")
        print(f"  Server broadcast:  {num_clients} × {global_model_size/1024:.1f} KB = {broadcast_bytes/1024:.1f} KB total")
        print(f"  Round total:       {total_round_bytes/1024:.1f} KB ({total_round_bytes/1024/1024:.3f} MB)")
        print(f"  Centralized equiv: {centralized_data_bytes/1024/1024:.1f} MB (all raw data)")
        print(f"  Savings vs central: {(1 - total_round_bytes/centralized_data_bytes)*100:.1f}%")

        self.comm_log.append({
            'round': server_round,
            'num_clients': num_clients,
            'per_client_upload_bytes': global_model_size,
            'total_upload_bytes': round_comm_bytes,
            'broadcast_bytes': broadcast_bytes,
            'total_round_bytes': total_round_bytes,
            'centralized_equiv_bytes': centralized_data_bytes,
            'savings_pct': (1 - total_round_bytes / centralized_data_bytes) * 100,
        })

        # Run poisoning detection
        print(f"\n{'='*60}")
        print(f"[Round {server_round}] Running poisoning detection...")
        print(f"[Round {server_round}] Received updates from {len(results)} clients")
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

    def save_comm_metrics(self):
        """Save communication metrics to CSV."""
        if not self.comm_log:
            return
        with open(COMM_METRICS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.comm_log[0].keys())
            writer.writeheader()
            writer.writerows(self.comm_log)

        # Print summary
        total_bytes = sum(r['total_round_bytes'] for r in self.comm_log)
        total_rounds = len(self.comm_log)
        centralized = self.comm_log[0]['centralized_equiv_bytes']

        print(f"\n{'='*60}")
        print(f"  COMMUNICATION METRICS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total rounds:          {total_rounds}")
        print(f"  Total FL comm:         {total_bytes/1024/1024:.3f} MB")
        print(f"  Centralized equiv:     {centralized/1024/1024:.1f} MB")
        print(f"  Overall savings:       {(1 - total_bytes/centralized)*100:.1f}%")
        print(f"  Avg per round:         {total_bytes/total_rounds/1024:.1f} KB")
        print(f"  Saved to:              {COMM_METRICS_FILE}")
        print(f"{'='*60}")


def start_server(num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS):
    print(f"\n{'='*60}")
    print(f"  Secure Federated Learning Server")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"  Poisoning threshold: {Z_THRESHOLD}")
    print(f"{'='*60}\n")

    model = create_model()
    initial_weights = model.get_weights()
    initial_parameters = ndarrays_to_parameters(initial_weights)

    try:
        blockchain = BlockchainHelper()
        num_accounts = len(blockchain.accounts)
        if num_accounts < num_clients + 1:
            print(f"WARNING: Ganache has {num_accounts} accounts but need {num_clients + 1}")
            print(f"Restart Ganache with: ganache --port 7545 --accounts {num_clients + 5}")
        print(f"Blockchain connected for FL server ({num_accounts} accounts available)")
    except Exception as e:
        print(f"Blockchain connection failed: {e}")
        blockchain = None

    detector = PoisoningDetector(z_threshold=Z_THRESHOLD)
    strategy = None

    if blockchain:
        strategy = SecureFedAvg(
            blockchain=blockchain,
            poisoning_detector=detector,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
        )

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Save communication metrics after training completes
    if isinstance(strategy, SecureFedAvg):
        strategy.save_comm_metrics()


if __name__ == '__main__':
    import sys
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_CLIENTS
    num_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_ROUNDS
    start_server(num_rounds=num_rounds, num_clients=num_clients)