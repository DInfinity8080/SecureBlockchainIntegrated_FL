import numpy as np
from typing import List, Tuple, Dict

class PoisoningDetector:
    def __init__(self, z_threshold=2.0):
        self.z_threshold = z_threshold
        self.history = []

    def compute_update_magnitude(self, weights: List[np.ndarray]) -> float:
        total = 0.0
        count = 0
        for w in weights:
            total += np.sum(np.abs(w))
            count += w.size
        return total / count

    def compute_update_direction(self, old_weights: List[np.ndarray], new_weights: List[np.ndarray]) -> np.ndarray:
        flat_old = np.concatenate([w.flatten() for w in old_weights])
        flat_new = np.concatenate([w.flatten() for w in new_weights])
        diff = flat_new - flat_old
        norm = np.linalg.norm(diff)
        if norm == 0:
            return diff
        return diff / norm

    def detect_poisoning(self, client_updates: List[Tuple[int, List[np.ndarray]]], 
                         global_weights: List[np.ndarray] = None) -> Dict[int, dict]:
        results = {}

        # Compute magnitude for each client
        magnitudes = {}
        for client_id, weights in client_updates:
            magnitudes[client_id] = self.compute_update_magnitude(weights)

        mag_values = list(magnitudes.values())
        mag_mean = np.mean(mag_values)
        mag_std = np.std(mag_values) if len(mag_values) > 1 else 1.0

        # Compute direction similarity if global weights provided
        directions = {}
        if global_weights is not None:
            for client_id, weights in client_updates:
                directions[client_id] = self.compute_update_direction(global_weights, weights)

        # Compute pairwise cosine similarities
        cosine_scores = {}
        if directions:
            client_ids = list(directions.keys())
            for cid in client_ids:
                sims = []
                for other_cid in client_ids:
                    if cid != other_cid:
                        cos_sim = np.dot(directions[cid], directions[other_cid])
                        sims.append(cos_sim)
                cosine_scores[cid] = np.mean(sims) if sims else 1.0

        cos_values = list(cosine_scores.values()) if cosine_scores else []
        cos_mean = np.mean(cos_values) if cos_values else 0
        cos_std = np.std(cos_values) if len(cos_values) > 1 else 1.0

        # Evaluate each client
        for client_id, weights in client_updates:
            mag = magnitudes[client_id]
            mag_z = abs(mag - mag_mean) / mag_std if mag_std > 0 else 0

            cos_z = 0
            if client_id in cosine_scores and cos_std > 0:
                cos_z = abs(cosine_scores[client_id] - cos_mean) / cos_std

            # Combined anomaly score (weighted average)
            anomaly_score = 0.6 * mag_z + 0.4 * cos_z
            is_poisoned = anomaly_score > self.z_threshold

            results[client_id] = {
                'magnitude': mag,
                'magnitude_z_score': mag_z,
                'cosine_similarity': cosine_scores.get(client_id, None),
                'cosine_z_score': cos_z,
                'anomaly_score': anomaly_score,
                'is_poisoned': is_poisoned,
                'passed_validation': not is_poisoned
            }

            status = "SUSPICIOUS" if is_poisoned else "CLEAN"
            print(f"[Client {client_id}] {status} | Anomaly: {anomaly_score:.4f} | "
                  f"Mag Z: {mag_z:.4f} | Cos Z: {cos_z:.4f}")

        self.history.append(results)
        return results

    def get_clean_clients(self, results: Dict[int, dict]) -> List[int]:
        return [cid for cid, info in results.items() if not info['is_poisoned']]

    def get_poisoned_clients(self, results: Dict[int, dict]) -> List[int]:
        return [cid for cid, info in results.items() if info['is_poisoned']]
