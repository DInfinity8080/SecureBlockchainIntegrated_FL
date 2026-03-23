"""
Centralized Baseline Training
==============================
Trains the same neural network on ALL data (no federation) to establish
a performance ceiling. Compare this against FL results to verify the
"within 5% accuracy" target.

Usage:
    python baseline.py
    python baseline.py 10    # train for 10 epochs
"""

from gpu_config import DEVICE_NAME  # auto-configures GPU/Metal/CPU
import os

import numpy as np
import csv
import time
from model import create_model
from data_loader import load_and_preprocess

RESULTS_FILE = "baseline_results.csv"


def train_centralized(num_epochs=15, batch_size=32):
    print("=" * 60)
    print("  Centralized Baseline Training")
    print("=" * 60)

    print("\nLoading NSL-KDD dataset...")
    X, y = load_and_preprocess()

    split = int(0.8 * len(X))
    x_train, x_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {np.unique(y)}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    print("-" * 60)

    model = create_model(input_dim=X.shape[1])

    results = []
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        history = model.fit(
            x_train, y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            verbose=0
        )

        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        val_loss = history.history['val_loss'][0]
        val_acc = history.history['val_accuracy'][0]
        epoch_time = time.time() - epoch_start

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch_time_s': epoch_time,
        })

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

    total_time = time.time() - start_time

    final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)

    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("  CENTRALIZED BASELINE RESULTS")
    print("=" * 60)
    print(f"  Final Test Loss:     {final_loss:.4f}")
    print(f"  Final Test Accuracy: {final_acc:.4f}")
    print(f"  Total Training Time: {total_time:.1f}s")
    print(f"  Results saved to:    {RESULTS_FILE}")
    print("=" * 60)

    print("\n  COMPARISON WITH FEDERATED LEARNING:")
    print("  -----------------------------------")
    print(f"  Centralized accuracy: {final_acc:.4f}")
    print(f"  FL target (within 5%): {final_acc - 0.05:.4f} - {final_acc:.4f}")
    print(f"  Check your FL server output for distributed loss/accuracy")
    print(f"  to verify FL performance is within this range.")
    print("=" * 60)

    return final_loss, final_acc, results


if __name__ == '__main__':
    import sys
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    train_centralized(num_epochs=num_epochs)
