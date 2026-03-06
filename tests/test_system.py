"""
Tests for Secure Blockchain-Integrated Federated Learning System
=================================================================
Run with: python -m pytest tests/ -v
Or:       python tests/test_system.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from data_loader import load_and_preprocess, partition_data, ATTACK_MAP, CLASS_MAP
from poisoning_detector import PoisoningDetector
from client import DEVICE_TIERS, assign_tier, compute_update_size


class TestModel(unittest.TestCase):
    def test_create_model_default(self):
        model = create_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 41))
        self.assertEqual(model.output_shape, (None, 5))

    def test_create_model_custom_input(self):
        model = create_model(input_dim=20, num_classes=3)
        self.assertEqual(model.input_shape, (None, 20))
        self.assertEqual(model.output_shape, (None, 3))

    def test_create_model_custom_lr(self):
        model = create_model(learning_rate=0.01)
        self.assertIsNotNone(model)

    def test_model_weights_shape(self):
        model = create_model(input_dim=41)
        weights = model.get_weights()
        self.assertEqual(len(weights), 8)
        self.assertEqual(weights[0].shape, (41, 128))
        self.assertEqual(weights[1].shape, (128,))

    def test_model_predict(self):
        model = create_model(input_dim=41)
        dummy_input = np.random.randn(5, 41).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        self.assertEqual(predictions.shape, (5, 5))
        for pred in predictions:
            self.assertAlmostEqual(np.sum(pred), 1.0, places=5)

    def test_model_train(self):
        model = create_model(input_dim=10, num_classes=3)
        x = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, size=100).astype(np.int32)
        history = model.fit(x, y, epochs=1, batch_size=32, verbose=0)
        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)


class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_and_preprocess()

    def test_data_loaded(self):
        self.assertIsNotNone(self.X)
        self.assertGreater(len(self.X), 0)

    def test_feature_count(self):
        self.assertEqual(self.X.shape[1], 41)

    def test_label_classes(self):
        unique_classes = set(np.unique(self.y))
        self.assertEqual(unique_classes, {0, 1, 2, 3, 4})

    def test_data_normalized(self):
        mean = np.mean(self.X, axis=0)
        self.assertTrue(np.all(np.abs(mean) < 1.0))

    def test_partition_count(self):
        for n in [3, 5, 10]:
            partitions = partition_data(self.X, self.y, num_clients=n)
            self.assertEqual(len(partitions), n)

    def test_partition_coverage(self):
        partitions = partition_data(self.X, self.y, num_clients=5)
        total = sum(len(p[0]) for p in partitions)
        self.assertEqual(total, len(self.X))

    def test_partition_nonempty(self):
        partitions = partition_data(self.X, self.y, num_clients=10)
        for px, py in partitions:
            self.assertGreater(len(px), 0)

    def test_attack_map_complete(self):
        for attack, category in ATTACK_MAP.items():
            self.assertIn(category, CLASS_MAP)


class TestPoisoningDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PoisoningDetector(z_threshold=1.5)

    def _make_weights(self, scale=1.0, seed=42):
        rng = np.random.RandomState(seed)
        return [
            rng.randn(41, 128).astype(np.float32) * scale,
            rng.randn(128).astype(np.float32) * scale,
            rng.randn(128, 64).astype(np.float32) * scale,
            rng.randn(64).astype(np.float32) * scale,
        ]

    def test_all_clean(self):
        updates = [(i, self._make_weights(1.0, seed=i)) for i in range(5)]
        results = self.detector.detect_poisoning(updates)
        clean = self.detector.get_clean_clients(results)
        self.assertGreater(len(clean), 0)

    def test_detect_outlier(self):
        """A client with extreme weights should be flagged."""
        updates = [(i, self._make_weights(1.0, seed=i)) for i in range(25)]
        updates.append((25, self._make_weights(1000.0, seed=25)))
        results = self.detector.detect_poisoning(updates)
        poisoned = self.detector.get_poisoned_clients(results)
        self.assertIn(25, poisoned)

    def test_results_structure(self):
        updates = [(i, self._make_weights(1.0, seed=i)) for i in range(2)]
        results = self.detector.detect_poisoning(updates)
        for cid, info in results.items():
            self.assertIn('magnitude', info)
            self.assertIn('anomaly_score', info)
            self.assertIn('is_poisoned', info)
            self.assertIn('passed_validation', info)

    def test_threshold_sensitivity(self):
        updates = [
            (0, self._make_weights(1.0, seed=0)),
            (1, self._make_weights(1.5, seed=1)),
            (2, self._make_weights(1.0, seed=2)),
            (3, self._make_weights(2.0, seed=3)),
            (4, self._make_weights(1.0, seed=4)),
        ]
        strict = PoisoningDetector(z_threshold=0.5)
        lenient = PoisoningDetector(z_threshold=3.0)
        strict_poisoned = strict.get_poisoned_clients(strict.detect_poisoning(updates))
        lenient_poisoned = lenient.get_poisoned_clients(lenient.detect_poisoning(updates))
        self.assertGreaterEqual(len(strict_poisoned), len(lenient_poisoned))

    def test_cosine_with_global_weights(self):
        global_w = self._make_weights(1.0, seed=99)
        updates = [(i, self._make_weights(1.0, seed=i)) for i in range(3)]
        results = self.detector.detect_poisoning(updates, global_weights=global_w)
        for cid, info in results.items():
            self.assertIsNotNone(info['cosine_similarity'])


class TestClientUtilities(unittest.TestCase):
    def test_tier_assignment(self):
        self.assertEqual(assign_tier(0, 10), 1)
        self.assertEqual(assign_tier(1, 10), 2)
        self.assertEqual(assign_tier(2, 10), 2)
        self.assertEqual(assign_tier(3, 10), 3)
        self.assertEqual(assign_tier(4, 10), 3)
        self.assertEqual(assign_tier(5, 10), 1)

    def test_all_tiers_valid(self):
        required = ['name', 'epochs', 'batch_size', 'data_fraction', 'learning_rate']
        for tier_id, config in DEVICE_TIERS.items():
            for field in required:
                self.assertIn(field, config)

    def test_compute_update_size(self):
        weights = [
            np.zeros((41, 128), dtype=np.float32),
            np.zeros((128,), dtype=np.float32),
        ]
        expected = 41 * 128 * 4 + 128 * 4
        self.assertEqual(compute_update_size(weights), expected)

    def test_tier_data_fractions(self):
        for tier_id, config in DEVICE_TIERS.items():
            self.assertGreater(config['data_fraction'], 0)
            self.assertLessEqual(config['data_fraction'], 1.0)

    def test_tier_epochs_ordering(self):
        self.assertGreater(DEVICE_TIERS[1]['epochs'], DEVICE_TIERS[2]['epochs'])
        self.assertGreater(DEVICE_TIERS[2]['epochs'], DEVICE_TIERS[3]['epochs'])


class TestBlockchainHelper(unittest.TestCase):
    def test_import(self):
        from blockchain_helper import BlockchainHelper
        self.assertIsNotNone(BlockchainHelper)

    def test_hash_weights(self):
        from blockchain_helper import BlockchainHelper
        weights = [np.ones((10, 10), dtype=np.float32)]
        bh = BlockchainHelper.__new__(BlockchainHelper)
        hash1 = bh._hash_weights(weights)
        hash2 = bh._hash_weights(weights)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_hash_different_weights(self):
        from blockchain_helper import BlockchainHelper
        w1 = [np.ones((10, 10), dtype=np.float32)]
        w2 = [np.zeros((10, 10), dtype=np.float32)]
        bh = BlockchainHelper.__new__(BlockchainHelper)
        self.assertNotEqual(bh._hash_weights(w1), bh._hash_weights(w2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
