import numpy as np
from typing import List, Tuple, Dict


def _mad_zscore(values: np.ndarray) -> np.ndarray:
    """Median Absolute Deviation (MAD) modified z-score.

    Robust against outliers even for very small N (≥2).  The constant 0.6745
    makes the score equivalent to a classical z-score under Gaussian data.
    Falls back to 0 when MAD == 0 (all values identical).
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        # All values are identical; no outliers possible
        return np.zeros_like(values, dtype=float)
    return 0.6745 * np.abs(values - median) / mad


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
        n = len(client_updates)

        # ── Magnitude scores (MAD-based, robust for small N) ─────────
        magnitudes = {}
        for client_id, weights in client_updates:
            magnitudes[client_id] = self.compute_update_magnitude(weights)

        mag_values = np.array([magnitudes[cid] for cid, _ in client_updates])
        mag_zscores = _mad_zscore(mag_values)
        mag_z_map = {cid: mag_zscores[i] for i, (cid, _) in enumerate(client_updates)}

        # ── Direction / cosine similarity scores ─────────────────────
        directions = {}
        if global_weights is not None:
            for client_id, weights in client_updates:
                directions[client_id] = self.compute_update_direction(global_weights, weights)

        cosine_scores = {}
        if directions:
            client_ids = list(directions.keys())
            for cid in client_ids:
                sims = [
                    np.dot(directions[cid], directions[other])
                    for other in client_ids if other != cid
                ]
                cosine_scores[cid] = float(np.mean(sims)) if sims else 1.0

        cos_values = np.array(list(cosine_scores.values())) if cosine_scores else np.array([])
        if len(cos_values) > 1:
            cos_zscores_arr = _mad_zscore(cos_values)
            cos_z_map = {cid: cos_zscores_arr[i] for i, cid in enumerate(cosine_scores)}
        else:
            cos_z_map = {cid: 0.0 for cid in (cosine_scores or {})}

        # ── Per-client anomaly decision ───────────────────────────────
        for client_id, weights in client_updates:
            mag = magnitudes[client_id]
            mag_z = mag_z_map.get(client_id, 0.0)
            cos_z = cos_z_map.get(client_id, 0.0)

            # Combined anomaly score (weighted average)
            anomaly_score = 0.6 * mag_z + 0.4 * cos_z

            # For very small cohorts (n < 5) raise the effective bar slightly
            # to compensate for reduced statistical power.
            effective_threshold = self.z_threshold * (1.0 + max(0, (5 - n) * 0.1))
            is_poisoned = anomaly_score > effective_threshold

            results[client_id] = {
                'magnitude': mag,
                'magnitude_z_score': mag_z,
                'cosine_similarity': cosine_scores.get(client_id, None),
                'cosine_z_score': cos_z,
                'anomaly_score': anomaly_score,
                'effective_threshold': round(effective_threshold, 4),
                'is_poisoned': is_poisoned,
                'passed_validation': not is_poisoned,
            }

            status = "SUSPICIOUS" if is_poisoned else "CLEAN"
            print(f"[Client {client_id}] {status} | Anomaly: {anomaly_score:.4f} | "
                  f"Mag Z: {mag_z:.4f} | Cos Z: {cos_z:.4f} "
                  f"(threshold: {effective_threshold:.2f}, n={n})")

        self.history.append(results)
        return results

    def get_clean_clients(self, results: Dict[int, dict]) -> List[int]:
        return [cid for cid, info in results.items() if not info['is_poisoned']]

    def get_poisoned_clients(self, results: Dict[int, dict]) -> List[int]:
        return [cid for cid, info in results.items() if info['is_poisoned']]
