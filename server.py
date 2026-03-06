import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
import numpy as np
import csv
import time
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Union
from model import create_model
from blockchain_helper import BlockchainHelper
from poisoning_detector import PoisoningDetector
from client import DEVICE_TIERS, TIER_DISTRIBUTION

# ── Configuration ────────────────────────────────────────────────
NUM_CLIENTS = 10
NUM_ROUNDS = 5
Z_THRESHOLD = 1.5
SERVER_ADDRESS = "0.0.0.0:9090"

# Dropout / Async settings
FRACTION_FIT = 0.7
FRACTION_EVALUATE = 1.0
MIN_FIT_CLIENTS = 3
MIN_EVALUATE_CLIENTS = 3
ROUND_TIMEOUT = 45

# Output directory
RESULTS_DIR = "results"
# ─────────────────────────────────────────────────────────────────


def compute_weights_size(weights):
    total = 0
    for w in weights:
        total += w.nbytes
    return total


def get_tier_for_client(client_id):
    return TIER_DISTRIBUTION[client_id % len(TIER_DISTRIBUTION)]


class SecureFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, blockchain, poisoning_detector, num_clients_expected, **kwargs):
        super().__init__(**kwargs)
        self.blockchain = blockchain
        self.detector = poisoning_detector
        self.global_weights = None
        self.client_account_map = {}
        self.num_clients_expected = num_clients_expected

        self.round_summary_log = []
        self.client_round_log = []
        self.poisoning_log = []
        self.reputation_history = []
        self.comm_log = []
        self.participation_log = []

        self.session_start = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.client_tiers = {}
        for i in range(num_clients_expected):
            tier = get_tier_for_client(i)
            self.client_tiers[i] = {
                'tier': tier,
                'tier_name': DEVICE_TIERS[tier]['name'],
                'epochs': DEVICE_TIERS[tier]['epochs'],
                'batch_size': DEVICE_TIERS[tier]['batch_size'],
                'data_fraction': DEVICE_TIERS[tier]['data_fraction'],
                'dropout_prob': DEVICE_TIERS[tier]['dropout_prob'],
            }

    def _register_devices(self, num_clients):
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

        round_start = time.time()

        # ── Filter out simulated dropouts (num_examples == 0) ────
        active_results = []
        simulated_drops = 0
        for client_proxy, fit_res in results:
            if fit_res.num_examples == 0:
                simulated_drops += 1
            else:
                active_results.append((client_proxy, fit_res))

        num_responded = len(active_results)
        num_failed = len(failures) + simulated_drops
        total_sampled = len(results) + len(failures)

        print(f"\n[Round {server_round}] Participation: "
              f"{num_responded}/{total_sampled} responded "
              f"({num_failed} dropped out)")

        if num_failed > 0:
            print(f"[Round {server_round}] WARNING: {num_failed} client(s) "
                  f"failed/timed out this round")

        self.participation_log.append({
            'session_id': self.session_id,
            'round': server_round,
            'sampled': total_sampled,
            'responded': num_responded,
            'dropped': num_failed,
            'participation_rate': round(num_responded / total_sampled, 4) if total_sampled > 0 else 0,
        })

        if not active_results:
            print(f"[Round {server_round}] All clients dropped! Skipping.")
            return None, {}

        # Register devices
        try:
            self._register_devices(len(active_results))
        except Exception as e:
            print(f"Device registration error: {e}")

        # Extract client updates
        client_updates = []
        client_info = []
        round_comm_bytes = 0

        for i, (client_proxy, fit_res) in enumerate(active_results):
            weights = parameters_to_ndarrays(fit_res.parameters)
            client_updates.append((i, weights))
            client_info.append((client_proxy, fit_res, i))

            update_size = compute_weights_size(weights)
            round_comm_bytes += update_size

        # ── Communication Metrics ────────────────────────────────
        global_model_size = compute_weights_size(
            parameters_to_ndarrays(active_results[0][1].parameters)
        )
        broadcast_bytes = global_model_size * num_responded
        total_round_bytes = round_comm_bytes + broadcast_bytes
        centralized_data_bytes = 148517 * 41 * 4

        self.comm_log.append({
            'session_id': self.session_id,
            'round': server_round,
            'num_clients': num_responded,
            'num_dropped': num_failed,
            'per_client_upload_bytes': global_model_size,
            'total_upload_bytes': round_comm_bytes,
            'broadcast_bytes': broadcast_bytes,
            'total_round_bytes': total_round_bytes,
            'centralized_equiv_bytes': centralized_data_bytes,
            'savings_pct': round((1 - total_round_bytes / centralized_data_bytes) * 100, 2),
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

        # Log per-client poisoning detection
        for client_id, info in detection_results.items():
            tier_info = self.client_tiers.get(client_id, {})
            self.poisoning_log.append({
                'session_id': self.session_id,
                'round': server_round,
                'client_id': client_id,
                'tier': tier_info.get('tier', 'N/A'),
                'tier_name': tier_info.get('tier_name', 'N/A'),
                'magnitude': round(info['magnitude'], 6),
                'magnitude_z_score': round(info['magnitude_z_score'], 4),
                'cosine_similarity': round(info['cosine_similarity'], 4) if info['cosine_similarity'] is not None else None,
                'cosine_z_score': round(info['cosine_z_score'], 4),
                'anomaly_score': round(info['anomaly_score'], 4),
                'is_poisoned': info['is_poisoned'],
                'passed_validation': info['passed_validation'],
            })

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

                tier_info = self.client_tiers.get(client_id, {})
                self.reputation_history.append({
                    'session_id': self.session_id,
                    'round': server_round,
                    'client_id': client_id,
                    'tier': tier_info.get('tier', 'N/A'),
                    'tier_name': tier_info.get('tier_name', 'N/A'),
                    'reputation': rep,
                    'passed_validation': info['passed_validation'],
                    'was_flagged': info['is_poisoned'],
                })
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

            tier_info = self.client_tiers.get(cid, {})
            self.client_round_log.append({
                'session_id': self.session_id,
                'round': server_round,
                'client_id': cid,
                'tier': tier_info.get('tier', 'N/A'),
                'tier_name': tier_info.get('tier_name', 'N/A'),
                'included_in_aggregation': True,
                'reputation_weight': round(weight_factor, 6),
                'reputation': rep_weights.get(cid, 100),
                'update_size_bytes': compute_weights_size(client_weights),
            })

        # Log excluded clients
        for client_proxy, fit_res, client_id in client_info:
            if client_id not in clean_ids:
                tier_info = self.client_tiers.get(client_id, {})
                self.client_round_log.append({
                    'session_id': self.session_id,
                    'round': server_round,
                    'client_id': client_id,
                    'tier': tier_info.get('tier', 'N/A'),
                    'tier_name': tier_info.get('tier_name', 'N/A'),
                    'included_in_aggregation': False,
                    'reputation_weight': 0.0,
                    'reputation': rep_weights.get(client_id, 100),
                    'update_size_bytes': compute_weights_size(
                        parameters_to_ndarrays(fit_res.parameters)),
                })

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

        round_time = time.time() - round_start

        self.round_summary_log.append({
            'session_id': self.session_id,
            'round': server_round,
            'num_clients_responded': num_responded,
            'num_clients_dropped': num_failed,
            'num_clean': len(clean_ids),
            'num_poisoned': len(poisoned_ids),
            'poisoned_client_ids': str(poisoned_ids),
            'total_comm_bytes': total_round_bytes,
            'round_time_s': round(round_time, 2),
        })

        return ndarrays_to_parameters(aggregated), {}

    def save_all_results(self, total_time, num_rounds, num_clients):
        os.makedirs(RESULTS_DIR, exist_ok=True)

        config = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'z_threshold': Z_THRESHOLD,
            'fraction_fit': FRACTION_FIT,
            'fraction_evaluate': FRACTION_EVALUATE,
            'min_fit_clients': MIN_FIT_CLIENTS,
            'round_timeout': ROUND_TIMEOUT,
            'total_time_s': round(total_time, 2),
        }
        with open(os.path.join(RESULTS_DIR, "session_config.json"), 'w') as f:
            json.dump(config, f, indent=2)

        tier_rows = []
        for cid in range(num_clients):
            info = self.client_tiers.get(cid, {})
            tier_rows.append({
                'session_id': self.session_id,
                'client_id': cid,
                'tier': info.get('tier', 'N/A'),
                'tier_name': info.get('tier_name', 'N/A'),
                'epochs': info.get('epochs', 'N/A'),
                'batch_size': info.get('batch_size', 'N/A'),
                'data_fraction': info.get('data_fraction', 'N/A'),
                'dropout_prob': info.get('dropout_prob', 'N/A'),
            })
        self._save_csv(os.path.join(RESULTS_DIR, "client_tiers.csv"), tier_rows)

        self._save_csv(os.path.join(RESULTS_DIR, "round_summary.csv"), self.round_summary_log)
        self._save_csv(os.path.join(RESULTS_DIR, "client_round_details.csv"), self.client_round_log)
        self._save_csv(os.path.join(RESULTS_DIR, "poisoning_detection.csv"), self.poisoning_log)
        self._save_csv(os.path.join(RESULTS_DIR, "reputation_history.csv"), self.reputation_history)
        self._save_csv(os.path.join(RESULTS_DIR, "communication_metrics.csv"), self.comm_log)
        self._save_csv(os.path.join(RESULTS_DIR, "participation_metrics.csv"), self.participation_log)

        rep_rows = []
        for client_id, account_idx in sorted(self.client_account_map.items()):
            try:
                rep = self.blockchain.get_reputation(account_index=account_idx)
            except:
                rep = "N/A"
            info = self.client_tiers.get(client_id, {})
            rep_rows.append({
                'session_id': self.session_id,
                'client_id': client_id,
                'tier': info.get('tier', 'N/A'),
                'tier_name': info.get('tier_name', 'N/A'),
                'final_reputation': rep,
            })
        self._save_csv(os.path.join(RESULTS_DIR, "final_reputations.csv"), rep_rows)

    def _save_csv(self, filepath, data):
        if not data:
            return
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def print_final_report(self, total_time=0):
        print("\n")
        print("╔" + "═" * 58 + "╗")
        print("║" + "  FEDERATED LEARNING — SESSION REPORT".center(58) + "║")
        print("║" + f"  Session: {self.session_id}".center(58) + "║")
        print("╠" + "═" * 58 + "╣")

        # Tier distribution
        print("║" + "  DEVICE HETEROGENEITY".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        tier_counts = {1: 0, 2: 0, 3: 0}
        for cid, info in self.client_tiers.items():
            tier_counts[info['tier']] = tier_counts.get(info['tier'], 0) + 1
        for t in [1, 2, 3]:
            name = DEVICE_TIERS[t]['name']
            count = tier_counts.get(t, 0)
            print(f"║  Tier {t} ({name}): {count}".ljust(59) + "║")

        # Training progress
        print("╠" + "═" * 58 + "╣")
        print("║" + "  TRAINING PROGRESS".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        total_rounds = len(self.round_summary_log)
        print(f"║  Rounds completed:         {total_rounds}".ljust(59) + "║")
        print(f"║  Total time:               {total_time:.1f}s".ljust(59) + "║")
        print("║" + " " * 58 + "║")
        for entry in self.round_summary_log:
            r = entry['round']
            n = entry['num_clients_responded']
            d = entry['num_clients_dropped']
            p = entry['num_poisoned']
            t = entry['round_time_s']
            parts = []
            if d > 0: parts.append(f"{d} dropped")
            if p > 0: parts.append(f"{p} flagged")
            extra = f" ({', '.join(parts)})" if parts else ""
            print(f"║  Round {r}: {n:>2} clients, {t:.1f}s{extra}".ljust(59) + "║")

        # Communication
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

        # Poisoning
        print("╠" + "═" * 58 + "╣")
        print("║" + "  POISONING DETECTION".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        total_flagged = sum(r['num_poisoned'] for r in self.round_summary_log)
        total_checks = sum(r['num_clients_responded'] for r in self.round_summary_log)
        print(f"║  Total checks:             {total_checks}".ljust(59) + "║")
        print(f"║  Total flagged:            {total_flagged}".ljust(59) + "║")
        if total_checks > 0:
            print(f"║  Flag rate:                {total_flagged/total_checks*100:.1f}%".ljust(59) + "║")
        print(f"║  Z-score threshold:        {Z_THRESHOLD}".ljust(59) + "║")

        # Reputation
        print("╠" + "═" * 58 + "╣")
        print("║" + "  BLOCKCHAIN REPUTATION".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        for client_id, account_idx in sorted(self.client_account_map.items()):
            try:
                rep = self.blockchain.get_reputation(account_index=account_idx)
                tier = self.client_tiers.get(client_id, {}).get('tier', '?')
                bar = "█" * (rep // 10)
                print(f"║  Client {client_id:>2} (T{tier}): {rep:>3}  {bar}".ljust(59) + "║")
            except:
                print(f"║  Client {client_id:>2}: N/A".ljust(59) + "║")

        # Files
        print("╠" + "═" * 58 + "╣")
        print("║" + "  RESEARCH DATA FILES".center(58) + "║")
        print("║" + "  " + "-" * 54 + "  ║")
        files = [
            "session_config.json", "client_tiers.csv",
            "round_summary.csv", "client_round_details.csv",
            "poisoning_detection.csv", "reputation_history.csv",
            "communication_metrics.csv", "participation_metrics.csv",
            "final_reputations.csv",
        ]
        for f in files:
            print(f"║  {RESULTS_DIR}/{f}".ljust(59) + "║")
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
            num_clients_expected=num_clients,
            fraction_fit=FRACTION_FIT,
            fraction_evaluate=FRACTION_EVALUATE,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
    fraction_fit=FRACTION_FIT,
    fraction_evaluate=FRACTION_EVALUATE,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_evaluate_clients=MIN_EVALUATE_CLIENTS,
    min_available_clients=MIN_FIT_CLIENTS,
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

    if isinstance(strategy, SecureFedAvg):
        strategy.save_all_results(total_time, num_rounds, num_clients)
        strategy.print_final_report(total_time=total_time)


if __name__ == '__main__':
    import sys
    num_clients = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_CLIENTS
    num_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_ROUNDS
    start_server(num_rounds=num_rounds, num_clients=num_clients)