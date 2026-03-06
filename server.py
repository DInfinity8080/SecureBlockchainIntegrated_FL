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

# Dropout / Async settings
FRACTION_FIT = 1.0          # Sample 100% of clients each round
FRACTION_EVALUATE = 1.0     # Evaluate on 100% of clients
MIN_FIT_CLIENTS = 3         # Minimum clients needed to proceed
MIN_EVALUATE_CLIENTS = 3
ROUND_TIMEOUT = 120         # Seconds to wait for clients per round
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
        self.comm_log = []
        self.participation_log = []

    def _register_devices(self, num_clients):
        """Register any new devices that haven't been registered yet."""
        new_registrations = 0
        for i in range(num_clients):
            if i not in self.client_account_map:
                account_idx = len(self.client_account_map) + 1
                self.client_account_map[i] = account_idx
                self.blockchain.ensure_registered(account_idx, f"fl_client_{i}")
                new_registrations += 1
        if new_registrations > 0:
            print(f"Registered {new_registrations} new devices on blockchain "
                  f"({len(self.client_account_map)} total)")

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            print(f"[Round {server_round}] No results received! Skipping round.")
            return None, {}

        num_responded = len(results)
        num_failed = len(failures)
        total_sampled = num_responded + num_failed

        # ── Dropout / Async Logging ──────────────────────────────
        print(f"\n[Round {server_round}] Participation: "
              f"{num_responded}/{total_sampled} responded "
              f"({num_failed} dropped out)")

        if num_failed > 0:
            print(f"[Round {server_round}] WARNING: {num_failed} client(s) "
                  f"failed/timed out this round")

        self.participation_log.append({
            'round': server_round,
            'sampled': total_sampled,
            'responded': num_responded,
            'dropped': num_failed,
            'participation_rate': num_responded / total_sampled if total_sampled > 0 else 0,
        })

        # Register devices
        try:
            self._register_devices(num_responded)
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

            update_size = compute_weights_size(weights)
            round_comm_bytes += update_size

        # ── Communication Metrics ────────────────────────────────
        global_model_size = compute_weights_size(
            parameters_to_ndarrays(results[0][1].parameters)
        )
        broadcast_bytes = global_model_size * num_responded
        total_round_bytes = round_comm_bytes + broadcast_bytes
        centralized_data_bytes = 148517 * 41 * 4

        self.comm_log.append({
            'round': server_round,
            'num_clients': num_responded,
            'num_dropped': num_failed,
            'per_client_upload_bytes': global_model_size,
            'total_upload_bytes': round_comm_bytes,
            'broadcast_bytes': broadcast_bytes,
            'total_round_bytes': total_round_bytes,
            'centralized_equiv_bytes': centralized_data_bytes,
            'savings_pct': (1 - total_round_bytes / centralized_data_bytes) * 100,
        })

        # Run poisoning detection
        print(f"\n[Round {server_round}] Running poisoning detection...")
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

        return ndarrays_to_parameters(aggregated), {}

    def save_comm_metrics(self):
        """Save communication metrics to CSV."""
        if not self.comm_log:
            return
        with open(COMM_METRICS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.comm_log[0].keys())
            writer.writeheader()
            writer.writerows(self.comm_log)

    def save_participation_log(self):
        """Save participation/dropout metrics to CSV."""
        if not self.participation_log:
            return
        filepath = "participation_metrics.csv"
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.participation_log[0].keys())
            writer.writeheader()
            writer.writerows(self.participation_log)

    def save_reputation_log(self):
        """Save final reputation scores to CSV."""
        filepath = "reputation_scores.csv"
        rows = []
        for client_id, account_idx in sorted(self.client_account_map.items()):
            try:
                rep = self.blockchain.get_reputation(account_index=account_idx)
            except:
                rep = "N/A"
            rows.append({
                'client_id': client_id,
                'account_index': account_idx,
                'final_reputation': rep,
            })
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['client_id', 'account_index', 'final_reputation'])
            writer.writeheader()
            writer.writerows(rows)

    def print_final_report(self, total_time=0):
        """Print a clean final report of the FL session."""
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + "  FEDERATED LEARNING — SESSION REPORT".center(58) + "║")
        print("╠" + "═" * 58 + "╣")

        # Training progress
        print("║" + "  TRAINING PROGRESS".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        total_rounds = len(self.comm_log)
        print(f"║  Rounds completed:         {total_rounds}".ljust(59) + "║")
        print(f"║  Total time:               {total_time:.1f}s".ljust(59) + "║")
        print("║" + " " * 58 + "║")
        for entry in self.comm_log:
            r = entry['round']
            n = entry['num_clients']
            d = entry['num_dropped']
            drop_str = f", {d} dropped" if d > 0 else ""
            print(f"║  Round {r}:  {n} clients{drop_str}".ljust(59) + "║")

        # Communication metrics
        print("╠" + "═" * 58 + "╣")
        print("║" + "  COMMUNICATION METRICS".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        total_bytes = sum(r['total_round_bytes'] for r in self.comm_log)
        centralized = self.comm_log[0]['centralized_equiv_bytes']
        per_client = self.comm_log[0]['per_client_upload_bytes']
        avg_round = total_bytes / len(self.comm_log)
        savings = (1 - total_bytes / centralized) * 100

        print(f"║  Per-client update size:   {per_client/1024:.1f} KB".ljust(59) + "║")
        print(f"║  Avg round total:          {avg_round/1024:.1f} KB".ljust(59) + "║")
        print(f"║  Total FL communication:   {total_bytes/1024/1024:.3f} MB".ljust(59) + "║")
        print(f"║  Centralized equivalent:   {centralized/1024/1024:.1f} MB".ljust(59) + "║")
        print(f"║  Bandwidth savings:        {savings:.1f}%".ljust(59) + "║")

        # Participation
        print("╠" + "═" * 58 + "╣")
        print("║" + "  PARTICIPATION & DROPOUT".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        avg_rate = np.mean([r['participation_rate'] for r in self.participation_log])
        total_drops = sum(r['dropped'] for r in self.participation_log)
        print(f"║  Avg participation rate:   {avg_rate*100:.1f}%".ljust(59) + "║")
        print(f"║  Total dropouts:           {total_drops}".ljust(59) + "║")

        # Reputation summary
        print("╠" + "═" * 58 + "╣")
        print("║" + "  BLOCKCHAIN REPUTATION".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        for client_id, account_idx in sorted(self.client_account_map.items()):
            try:
                rep = self.blockchain.get_reputation(account_index=account_idx)
                bar = "█" * (rep // 10)
                print(f"║  Client {client_id:>2}:  {rep:>3}  {bar}".ljust(59) + "║")
            except:
                print(f"║  Client {client_id:>2}:  N/A".ljust(59) + "║")
        print(f"║  (Full details in reputation_scores.csv)".ljust(59) + "║")

        # Files saved
        print("╠" + "═" * 58 + "╣")
        print("║" + "  FILES SAVED".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        print("║  communication_metrics.csv".ljust(59) + "║")
        print("║  participation_metrics.csv".ljust(59) + "║")
        print("║  reputation_scores.csv".ljust(59) + "║")
        print("╚" + "═" * 58 + "╝")
        print()


def start_server(num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS):
    print(f"\n{'='*60}")
    print(f"  Secure Federated Learning Server")
    print(f"  Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"  Fraction fit: {FRACTION_FIT} | Min clients: {MIN_FIT_CLIENTS}")
    print(f"  Round timeout: {ROUND_TIMEOUT}s")
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
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=FRACTION_EVALUATE,
            min_fit_clients=MIN_FIT_CLIENTS,
            min_evaluate_clients=MIN_EVALUATE_CLIENTS,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=FRACTION_EVALUATE,
            min_fit_clients=MIN_FIT_CLIENTS,
            min_evaluate_clients=MIN_EVALUATE_CLIENTS,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
        )

    start_time = time.time()

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(
            num_rounds=num_rounds,
            round_timeout=ROUND_TIMEOUT,
        ),
        strategy=strategy,
    )

    total_time = time.time() - start_time

    # Save metrics and print report after training completes
    if isinstance(strategy, SecureFedAvg):
        strategy.save_comm_metrics()
        strategy.save_participation_log()
        strategy.save_reputation_log()
        strategy.print_final_report(total_time=total_time)


if __name__ == '__main__':
    import sys
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_CLIENTS
    num_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_ROUNDS
    start_server(num_rounds=num_rounds, num_clients=num_clients)