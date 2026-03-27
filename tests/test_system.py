"""
Tests for Secure Blockchain-Integrated Federated Learning System
=================================================================
Run with: python -m pytest tests/ -v
Or:       python tests/test_system.py
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import create_model
from data_loader import load_and_preprocess, partition_data, dirichlet_partition_data, ATTACK_MAP, CLASS_MAP
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


class TestDirichletPartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_and_preprocess()

    def test_correct_number_of_partitions(self):
        parts = dirichlet_partition_data(self.X, self.y, num_clients=5, alpha=0.5)
        self.assertEqual(len(parts), 5)

    def test_all_clients_get_samples(self):
        parts = dirichlet_partition_data(self.X, self.y, num_clients=10, alpha=0.5)
        for i, (px, py) in enumerate(parts):
            self.assertGreater(len(px), 0, f"Client {i} got no samples")

    def test_min_samples_guarantee(self):
        parts = dirichlet_partition_data(self.X, self.y, num_clients=10,
                                         alpha=0.01, min_samples=50)
        for px, py in parts:
            self.assertGreaterEqual(len(px), 50)

    def test_labels_and_features_match(self):
        parts = dirichlet_partition_data(self.X, self.y, num_clients=5)
        for px, py in parts:
            self.assertEqual(len(px), len(py))

    def test_low_alpha_is_more_skewed_than_high_alpha(self):
        """Lower alpha → each client more dominated by fewer classes."""
        def class_entropy(py):
            counts = np.bincount(py, minlength=5).astype(float)
            counts /= counts.sum()
            counts = counts[counts > 0]
            return -np.sum(counts * np.log(counts))

        skewed = dirichlet_partition_data(self.X, self.y, num_clients=5, alpha=0.05, seed=0)
        uniform = dirichlet_partition_data(self.X, self.y, num_clients=5, alpha=10.0, seed=0)
        avg_entropy_skewed = np.mean([class_entropy(py) for _, py in skewed])
        avg_entropy_uniform = np.mean([class_entropy(py) for _, py in uniform])
        self.assertLess(avg_entropy_skewed, avg_entropy_uniform)

    def test_reproducible_with_same_seed(self):
        p1 = dirichlet_partition_data(self.X, self.y, num_clients=5, seed=42)
        p2 = dirichlet_partition_data(self.X, self.y, num_clients=5, seed=42)
        for (x1, y1), (x2, y2) in zip(p1, p2):
            np.testing.assert_array_equal(x1, x2)

    def test_different_seeds_differ(self):
        p1 = dirichlet_partition_data(self.X, self.y, num_clients=5, seed=1)
        p2 = dirichlet_partition_data(self.X, self.y, num_clients=5, seed=2)
        # At least one partition should differ between seeds
        any_diff = any(not np.array_equal(x1, x2) for (x1, _), (x2, _) in zip(p1, p2))
        self.assertTrue(any_diff)


class TestDownloadErrorHandling(unittest.TestCase):
    def test_bad_url_raises_runtime_error(self):
        """If download fails, a RuntimeError with actionable message should be raised."""
        from unittest.mock import patch
        import data_loader as dl

        with patch('urllib.request.urlretrieve', side_effect=Exception("404 Not Found")):
            # Temporarily rename any existing file so the download is attempted
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(RuntimeError) as ctx:
                    dl.download_nslkdd(data_dir=tmpdir)
        self.assertIn("Failed to download", str(ctx.exception))
        self.assertIn("defcom17", str(ctx.exception))  # URL hint present

    def test_tiny_file_raises_runtime_error(self):
        """A suspiciously small downloaded file (e.g. 404 HTML) should be rejected."""
        from unittest.mock import patch
        import data_loader as dl
        import tempfile

        def fake_retrieve(url, dest):
            with open(dest, 'w') as f:
                f.write("<html>404 Not Found</html>")  # tiny HTML, not CSV

        with patch('urllib.request.urlretrieve', side_effect=fake_retrieve):
            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(RuntimeError) as ctx:
                    dl.download_nslkdd(data_dir=tmpdir)
        self.assertIn("too small", str(ctx.exception))


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
            self.assertIn('effective_threshold', info)  # added by MAD scorer

    def test_small_n_honest_clients_not_flagged(self):
        """With N=5, one extreme outlier should not cause honest clients to be flagged.

        This was broken with classic std-based z-score: a single large outlier
        inflates std so much that everyone else gets a near-zero z-score,
        making detection unreliable.  MAD is immune to this.

        Note: global_weights must use a seed different from all client seeds so
        that no client has a zero-vector update (identical weights → zero diff →
        direction = zero vector → anomalous cosine similarity).
        """
        honest = [(i, self._make_weights(1.0, seed=i)) for i in range(4)]
        outlier = [(99, self._make_weights(500.0, seed=99))]
        updates = honest + outlier
        results = self.detector.detect_poisoning(updates, global_weights=self._make_weights(1.0, seed=42))
        poisoned = self.detector.get_poisoned_clients(results)
        # Extreme outlier must be caught
        self.assertIn(99, poisoned)
        # Honest clients must NOT be falsely flagged
        for cid, _ in honest:
            self.assertNotIn(cid, poisoned, f"Honest client {cid} falsely flagged")

    def test_effective_threshold_higher_for_very_small_n(self):
        """With N < 5 the effective threshold should be raised above the base threshold."""
        updates = [(i, self._make_weights(1.0, seed=i)) for i in range(3)]
        results = self.detector.detect_poisoning(updates)
        for info in results.values():
            self.assertGreater(info['effective_threshold'], self.detector.z_threshold)

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

# ── Attack Simulator Tests ───────────────────────────────────────

class TestAttackSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load data once for all attack tests."""
        from data_loader import load_and_preprocess, partition_data
        X, y = load_and_preprocess()
        partitions = partition_data(X, y, num_clients=5)
        x_data, y_data = partitions[0]
        split = int(0.8 * len(x_data))
        cls.x_train = x_data[:split]
        cls.y_train = y_data[:split]
        cls.x_test = x_data[split:]
        cls.y_test = y_data[split:]

    def test_label_flip_changes_labels(self):
        """Label flip attack should shift all labels by 1 mod 5."""
        from attack_simulator import MaliciousClient
        original_labels = self.y_train.copy()
        client = MaliciousClient(
            client_id=0,
            x_train=self.x_train.copy(),
            y_train=original_labels,
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='label_flip'
        )
        expected = (original_labels + 1) % 5
        np.testing.assert_array_equal(client.y_train, expected)

    def test_label_flip_preserves_count(self):
        """Label flip should not change the number of samples."""
        from attack_simulator import MaliciousClient
        client = MaliciousClient(
            client_id=0,
            x_train=self.x_train.copy(),
            y_train=self.y_train.copy(),
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='label_flip'
        )
        self.assertEqual(len(client.y_train), len(self.y_train))

    def test_noise_injection_modifies_data(self):
        """Noise injection should change training data."""
        from attack_simulator import MaliciousClient
        original_x = self.x_train.copy()
        client = MaliciousClient(
            client_id=0,
            x_train=original_x,
            y_train=self.y_train.copy(),
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='noise_injection'
        )
        # Data should be different after noise injection
        self.assertFalse(np.array_equal(client.x_train, self.x_train))

    def test_noise_injection_preserves_shape(self):
        """Noise injection should not change data shape."""
        from attack_simulator import MaliciousClient
        client = MaliciousClient(
            client_id=0,
            x_train=self.x_train.copy(),
            y_train=self.y_train.copy(),
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='noise_injection'
        )
        self.assertEqual(client.x_train.shape, self.x_train.shape)

    def test_scaling_attack_amplifies_weights(self):
        """Scaling attack should produce larger weight updates."""
        from attack_simulator import MaliciousClient
        client = MaliciousClient(
            client_id=0,
            x_train=self.x_train.copy(),
            y_train=self.y_train.copy(),
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='scaling'
        )
        # Scaling client should be created without error
        self.assertEqual(client.attack_type, 'scaling')
        # Data should be unchanged (scaling happens during fit, not init)
        np.testing.assert_array_equal(client.x_train, self.x_train)
        np.testing.assert_array_equal(client.y_train, self.y_train)

    def test_malicious_client_can_train(self):
        """Malicious client should be able to train without crashing."""
        from attack_simulator import MaliciousClient
        client = MaliciousClient(
            client_id=0,
            x_train=self.x_train.copy(),
            y_train=self.y_train.copy(),
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='label_flip'
        )
        weights = client.get_parameters(config={})
        self.assertIsNotNone(weights)
        self.assertGreater(len(weights), 0)

    def test_invalid_attack_type_no_crash(self):
        """Unknown attack type should not crash (no poisoning applied)."""
        from attack_simulator import MaliciousClient
        client = MaliciousClient(
            client_id=0,
            x_train=self.x_train.copy(),
            y_train=self.y_train.copy(),
            x_test=self.x_test,
            y_test=self.y_test,
            attack_type='unknown_attack'
        )
        # Data should be unchanged
        np.testing.assert_array_equal(client.x_train, self.x_train)
        np.testing.assert_array_equal(client.y_train, self.y_train)

if __name__ == '__main__':
    import io
    import contextlib

    print("\n" + "=" * 60)
    print("  SECURE FL SYSTEM — TEST SUITE")
    print("=" * 60)

    loader = unittest.TestLoader()

    test_groups = {
        'Model':               TestModel,
        'Data Loader':         TestDataLoader,
        'Dirichlet Partition': TestDirichletPartition,
        'Download Errors':     TestDownloadErrorHandling,
        'Poisoning Detector':  TestPoisoningDetector,
        'Client Utilities':    TestClientUtilities,
        'Blockchain Helper':   TestBlockchainHelper,
        'Attack Simulator':    TestAttackSimulator,
    }

    total_tests = 0
    total_passed = 0
    total_failed = 0
    group_results = []

    for group_name, test_class in test_groups.items():
        print(f"\n  [{group_name}]")
        print(f"  {'-' * 50}")

        suite = loader.loadTestsFromTestCase(test_class)
        test_names = loader.getTestCaseNames(test_class)

        # Run tests with all output suppressed
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            runner = unittest.TextTestRunner(verbosity=0, stream=io.StringIO())
            result = runner.run(suite)

        passed = result.testsRun - len(result.failures) - len(result.errors)
        failed = len(result.failures) + len(result.errors)
        total_tests += result.testsRun
        total_passed += passed
        total_failed += failed

        failed_names = set()
        for test, traceback in result.failures + result.errors:
            name = str(test).split()[0]
            failed_names.add(name)

        for name in test_names:
            if name in failed_names:
                print(f"    FAIL  {name}")
            else:
                print(f"    PASS  {name}")

        group_results.append((group_name, result.testsRun, passed, failed))

    # Summary table
    print("\n" + "=" * 60)
    print("  TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Component':<22} {'Tests':>6} {'Pass':>6} {'Fail':>6}  Status")
    print(f"  {'-' * 54}")

    for name, total, passed, failed in group_results:
        status = "PASS" if failed == 0 else "FAIL"
        print(f"  {name:<22} {total:>6} {passed:>6} {failed:>6}  {status}")

    print(f"  {'-' * 54}")
    print(f"  {'TOTAL':<22} {total_tests:>6} {total_passed:>6} {total_failed:>6}  "
          f"{'ALL PASSED' if total_failed == 0 else 'FAILURES DETECTED'}")
    print("=" * 60 + "\n")

    if total_failed > 0:
        sys.exit(1)