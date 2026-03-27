# Secure Blockchain-Integrated Federated Learning for IoT Edge Devices

A privacy-preserving distributed machine learning system that combines **Federated Learning** with **Ethereum Blockchain** technology to enable secure, verifiable, and Byzantine-resilient model training across heterogeneous IoT edge devices. The system performs network intrusion detection using the NSL-KDD dataset, with z-score-based poisoning detection, on-chain reputation management, and reputation-weighted aggregation — all validated through **330+ individual experiments** across 66 configurations.

> **Research Project**
> Kapilrajsinh Jadeja, Dev Vinay Patel, Laeya Royan Sadhya  
> Guided by: Dr. Saeed Sameet | Mentored by: Shahin Zanbaghi  
> March 2026

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           UBUNTU SERVER                              │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐   │
│  │   GANACHE     │   │   TRUFFLE    │   │    FLOWER SERVER       │   │
│  │   Local ETH   │   │   Smart      │   │    FL Coordinator      │   │
│  │   Blockchain  │   │   Contract   │   │    SecureFedAvg        │   │
│  │   Port: 7545  │   │   Solidity   │   │    Port: 9090          │   │
│  └──────┬───────┘   └──────────────┘   └──────────┬─────────────┘   │
│         │         Blockchain Verification          │  FL Protocol    │
└─────────┼──────────────────────────────────────────┼────────────────┘
          │                                          │
    ┌─────┴──────┬──────────┬──────────┬─────────────┴────┐
    ▼            ▼          ▼          ▼                   ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────────┐
│Client 0│ │Client 1│ │Client 2│ │Client 3│ │  Client 4 .. N   │
│ Tier 1 │ │ Tier 2 │ │ Tier 2 │ │ Tier 3 │ │   Mixed Tiers    │
│5 epochs│ │3 epochs│ │3 epochs│ │1 epoch │ │   Configurable   │
│100% dat│ │70% data│ │70% data│ │40% data│ │   Per-tier       │
└────────┘ └────────┘ └────────┘ └────────┘ └──────────────────┘

Dataset: NSL-KDD (Network Intrusion Detection) | 41 Features | 5 Classes
Clients: 10–20 supported | Device Tiers: 3 | Dropout: Simulated (0–25%)
```

### Three-Layer Design

| Layer | Components | Responsibility |
|-------|-----------|----------------|
| **IoT Device Layer** | `client.py`, `attack_simulator.py` | Local training, tier-based heterogeneity, dropout simulation |
| **Federated Aggregation Layer** | `server.py`, `poisoning_detector.py` | SecureFedAvg, z-score anomaly detection, reputation-weighted aggregation |
| **Blockchain Layer** | `FederatedLearning.sol`, `blockchain_helper.py` | Immutable audit trail, on-chain reputation, SHA-256 model hashing |

---

## Key Features

- **Federated Learning with Flower**: Distributed model training using the Flower framework with FedAvg aggregation — raw data never leaves individual devices
- **Ethereum Blockchain Integration**: Immutable audit trail of all model updates, validations, and global model hashes via Solidity smart contracts on Ganache
- **Z-Score Dual-Metric Poisoning Detection**: Anomaly detection using weight magnitude (60%) and cosine similarity (40%) with a configurable z-score threshold (default: 1.5)
- **Blockchain-Managed Reputation System**: Trust scores stored on-chain that reward honest participants (+10, max 200) and penalize flagged ones (−20, min 10), creating asymmetric accountability
- **Reputation-Weighted Aggregation**: Clean clients contribute to the global model proportional to their reputation, reducing the influence of less trusted devices
- **3-Tier Device Heterogeneity**: Simulates real IoT deployments with Powerful Edge Servers (Tier 1), Mid-Range IoT Devices (Tier 2), and Constrained Sensor Nodes (Tier 3)
- **Dropout and Async Handling**: Clients simulate connection failures based on tier reliability; the server gracefully continues with available clients (non-blocking, zero-sleep design)
- **Attack Simulation Suite**: Built-in label flipping, noise injection (σ=2.0), and gradient scaling (10×) attacks for security testing
- **Communication Efficiency**: 90–95% bandwidth savings vs. centralized training, with per-round metrics tracking
- **Comprehensive Research Pipeline**: Exports 9 CSV files + 1 JSON config per session for reproducible analysis
- **Hardware Auto-Detection**: `gpu_config.py` detects Metal (Apple Silicon), CUDA, ROCm, or falls back to CPU
- **Automated Testing**: 34 unit tests across 6 components with 100% pass rate

---

## Experimental Results

### Summary of Findings (330+ Experiments)

Scalability experiments were conducted across **66 configurations** (11 client counts × 3 round durations × 2 conditions), producing 330+ individual training sessions.

| Metric | Result |
|--------|--------|
| **Final Accuracy Range** | 96.7% – 98.6% across all configurations |
| **Best Configuration** | 11 clients, 7 rounds: **98.28%** accuracy |
| **Centralized Baseline** | 81.1% (15 epochs) — FL outperforms by +16pp |
| **Bandwidth Savings** | 90–95% vs. centralized equivalent |
| **Convergence** | Near-optimal accuracy within 3–4 rounds |
| **Poisoning Detection Rate** | 8–17% flag rate across scales |
| **Mean Reputation** | 128–175 (out of 200 max) |
| **Participation Rate (dropout)** | 83–97% |
| **Participation Rate (no dropout)** | 100% |

### Accuracy Scalability (No Dropout)

| Clients | 5 Rounds | 7 Rounds | 10 Rounds |
|---------|----------|----------|-----------|
| 10 | 97.42% | 98.28% | 98.02% |
| 12 | 97.71% | 98.09% | 97.76% |
| 14 | 97.73% | 97.82% | 97.92% |
| 16 | 97.73% | 97.72% | 97.98% |
| 18 | 97.70% | 97.71% | 98.02% |
| 20 | 96.69% | 97.11% | 97.88% |

### Dropout Resilience

| Metric | With Dropout (30%) | No Dropout (Ideal) |
|--------|-------------------|-------------------|
| Accuracy | 97.9% | 97.8% |
| Reputation Score | 144.9 | 146.3 |
| Detection Rate | 7.4% | 11.0% |
| Participation Rate | 86.9% | 100.0% |

**Key finding**: Only −0.14% accuracy loss despite 30% client dropout probability, validating the system's robustness under realistic failure conditions.

### Representative Session (10 Clients, 5 Rounds, Dropout Enabled)

| Round | Responded | Dropped | Flagged | Accuracy | Loss | Time |
|-------|-----------|---------|---------|----------|------|------|
| 1 | 9 | 1 | 0 | 94.74% | 0.1637 | 0.27s |
| 2 | 9 | 1 | 2 | 96.86% | 0.0958 | 0.21s |
| 3 | 8 | 2 | 0 | 97.55% | 0.0792 | 0.23s |
| 4 | 8 | 2 | 0 | 97.62% | 0.0723 | 0.23s |
| 5 | 9 | 1 | 2 | 97.84% | 0.0680 | 0.21s |

Final reputations: 110–150 | Communication: 5.21 MB vs. 23.2 MB centralized (77.6% savings)

---

## Neural Network Model

A fully-connected feedforward network for 5-class network intrusion detection:

| # | Layer | Units/Rate | Activation | Parameters |
|---|-------|-----------|------------|------------|
| 1 | Dense | 128 | ReLU | 5,376 |
| 2 | Dropout | 0.3 | — | 0 |
| 3 | Dense | 64 | ReLU | 8,256 |
| 4 | Dropout | 0.3 | — | 0 |
| 5 | Dense | 32 | ReLU | 2,080 |
| 6 | Dense (output) | 5 | Softmax | 165 |

**Total**: 15,877 trainable parameters | **Update size**: ~62 KB per client per round

---

## Poisoning Detection

The system uses a dual-metric z-score approach to detect Byzantine participants:

1. **Weight Magnitude Analysis** (60% weight): Computes the mean absolute value of each client's model weights — catches abnormally large or scaled-up updates
2. **Cosine Similarity Analysis** (40% weight): Measures directional alignment between each client's update and all other clients — catches updates pointing away from the consensus

**Combined anomaly score**: `score = 0.6 × z_mag + 0.4 × z_cos`

A score exceeding the threshold (default: 1.5) flags the client as suspicious. Flagged clients are excluded from aggregation and their blockchain reputation drops by 20 points.

### Reputation System

| Event | Reputation Change | Bounds |
|-------|------------------|--------|
| Clean validation | +10 | Max 200 |
| Flagged as poisoned | −20 | Min 10 |
| Initial registration | 100 | — |

**Asymmetric penalty**: One flag requires two consecutive clean rounds to fully recover, creating strong incentives for honest participation.

---

## Device Tiers

| Tier | Device Type | Epochs | Batch Size | Data Fraction | Learning Rate | Dropout Prob |
|------|-------------|--------|------------|---------------|---------------|-------------|
| 1 | Powerful Edge Server | 5 | 16 | 100% | 0.001 | 0% |
| 2 | Mid-Range IoT Device | 3 | 32 | 70% | 0.001 | 10% |
| 3 | Constrained Sensor Node | 1 | 64 | 40% | 0.0005 | 25% |

**Distribution pattern**: `[T1, T2, T2, T3, T3]` repeating → ~20% Tier 1, ~40% Tier 2, ~40% Tier 3

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Federated Learning | [Flower](https://flower.dev/) (flwr) | FL coordination, FedAvg, gRPC communication |
| Machine Learning | [TensorFlow](https://www.tensorflow.org/) / Keras | Neural network model, local training |
| Blockchain | [Ganache](https://trufflesuite.com/ganache/) + [Truffle](https://trufflesuite.com/) | Local Ethereum blockchain simulation |
| Smart Contracts | Solidity | Device registration, reputation ledger, audit trail |
| Blockchain Client | Web3.py | Python-to-Ethereum interaction |
| Dataset | [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) | Network intrusion detection (148,517 samples, 41 features, 5 classes) |
| Hardware Config | gpu_config.py | Auto-detection: Metal, CUDA, ROCm, CPU fallback |

---

## Project Structure

```
fl-project/
├── server.py                    # Flower server with SecureFedAvg strategy
├── client.py                    # Flower client with tier heterogeneity
├── model.py                     # TensorFlow/Keras neural network (Dense + Dropout)
├── data_loader.py               # NSL-KDD download, preprocessing, Non-IID partitioning
├── poisoning_detector.py        # Z-score dual-metric anomaly detection
├── blockchain_helper.py         # Web3.py blockchain interaction wrapper
├── attack_simulator.py          # Malicious client: label flip, noise, scaling
├── baseline.py                  # Centralized training baseline for comparison
├── gpu_config.py                # Hardware auto-detection (Metal, CUDA, ROCm, CPU)
├── contracts/
│   └── FederatedLearning.sol    # Solidity smart contract
├── migrations/
│   ├── 1_initial_migration.js
│   └── 2_deploy_contracts.js    # Contract deployment script
├── tests/
│   └── test_system.py           # 34 automated tests across 6 components
├── results/                     # Research data output (9 CSVs + 1 JSON per session)
├── logs/                        # Client and server logs
├── build/                       # Compiled contract artifacts
├── data/                        # NSL-KDD dataset (auto-downloaded)
├── run.sh                       # One-command full system launcher
├── stop.sh                      # Kill all running processes
├── launch_clients.sh            # Launch N clients in background
├── requirements.txt             # Python dependencies
├── truffle-config.js            # Truffle network configuration
└── README.md
```

### Codebase Layers

| File | Layer | Purpose |
|------|-------|---------|
| `server.py` | Aggregation | FL server, SecureFedAvg, reputation-weighted aggregation, CSV export |
| `client.py` | IoT Device | FL client, tier-based heterogeneity, dropout simulation, local training |
| `poisoning_detector.py` | Security | Z-score dual-metric anomaly detection (magnitude + cosine) |
| `blockchain_helper.py` | Blockchain | Web3.py wrapper: device registration, validation, reputation queries |
| `FederatedLearning.sol` | Blockchain | Solidity contract: device registry, reputation ledger, audit trail |
| `model.py` | ML Core | TensorFlow/Keras neural network (Dense + Dropout layers) |
| `data_loader.py` | Data | NSL-KDD download, preprocessing, StandardScaler, Non-IID partitioning |
| `attack_simulator.py` | Security | MaliciousClient: label flip, noise injection, gradient scaling |
| `baseline.py` | Evaluation | Centralized training baseline for accuracy comparison |
| `gpu_config.py` | Infrastructure | Hardware auto-detection (Metal, CUDA, ROCm, CPU) |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/DInfinity8080/SecureBlockchainIntegrated_FL.git
cd SecureBlockchainIntegrated_FL
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install Truffle & Ganache

```bash
# With sudo access:
sudo npm install -g truffle ganache

# Without sudo:
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
npm install -g truffle ganache
```

---

## Usage

### Quick Start (One Command)

```bash
# 10 clients, 5 rounds, dropout enabled
./run.sh 10 5

# 10 clients, 5 rounds, dropout disabled
./run.sh 10 5 nodrop

# 20 clients, 10 rounds
./run.sh 20 10

# Stop everything
./stop.sh
```

### Manual Launch (3 Terminals)

**Terminal 1** — Ganache Blockchain:
```bash
ganache --port 7545 --accounts 25
```

**Terminal 2** — Deploy Contracts + Start FL Server:
```bash
source venv/bin/activate
truffle migrate --reset --network development
python server.py 10 5
```

**Terminal 3** — Launch Clients:
```bash
./launch_clients.sh 10
```

### Running Attack Simulations

Replace one or more honest clients with malicious ones:

```bash
# Label flip attack (shifts all labels by 1)
python attack_simulator.py 0 label_flip

# Noise injection attack (Gaussian noise, σ=2.0)
python attack_simulator.py 0 noise_injection

# Scaling attack (amplifies weight updates 10×)
python attack_simulator.py 0 scaling
```

**Example — 1 attacker + 9 honest clients:**

| Terminal | Command | Role |
|----------|---------|------|
| 1 | `ganache --port 7545 --accounts 25` | Blockchain |
| 2 | `python server.py 10 5` | FL Server |
| 3 | `python attack_simulator.py 0 label_flip` | Attacker |
| 4–11 | `python client.py 1` through `python client.py 9` | Honest |

### Centralized Baseline

```bash
python baseline.py        # 15 epochs default
python baseline.py 20     # custom epochs
```

### Running Tests

```bash
python tests/test_system.py
```

```
============================================================
  TEST RESULTS SUMMARY
============================================================
  Component               Tests   Pass   Fail  Status
  ------------------------------------------------------
  Model                       6      6      0  PASS
  Data Loader                 8      8      0  PASS
  Poisoning Detector          5      5      0  PASS
  Client Utilities            5      5      0  PASS
  Blockchain Helper           3      3      0  PASS
  Attack Simulator            7      7      0  PASS
  ------------------------------------------------------
  TOTAL                      34     34      0  ALL PASSED
============================================================
```

### CLI Reference

```bash
# Server
python server.py [num_clients] [num_rounds]

# Client
python client.py [client_id] [num_clients] [flags]
python client.py 0 20 --no-dropout --random-tiers

# Attack Simulator
python attack_simulator.py [client_id] [attack_type] [num_clients]
python attack_simulator.py 0 label_flip 20

# Baseline
python baseline.py [num_epochs]

# Full System
./run.sh [num_clients] [num_rounds] [nodrop]
```

---

## Smart Contract

The `FederatedLearning.sol` contract manages four key functions:

| Function | Description |
|----------|-------------|
| **Device Registration** | Each IoT device registers with a unique ID and receives initial reputation of 100 |
| **Model Update Tracking** | SHA-256 hashes of model weights stored on-chain with timestamps and accuracy |
| **Validation & Reputation** | Server validates each update; reputation +10 (pass, max 200) or −20 (fail, min 10) |
| **Global Model Recording** | Each round's aggregated global model hash is immutably stored |

### Events Emitted

- `DeviceRegistered(address, deviceId)`
- `ModelUpdateSubmitted(address, round, modelHash)`
- `ModelValidated(address, round, passed)`
- `GlobalModelUpdated(round, modelHash)`
- `ReputationUpdated(address, newReputation)`

---

## Research Data Output

Each session exports **9 CSV files + 1 JSON config** to the `results/` directory:

| File | Contents |
|------|----------|
| `session_config.json` | Run settings: clients, rounds, thresholds, timeout |
| `client_tiers.csv` | Per-client tier assignment, epochs, batch size, data fraction |
| `round_summary.csv` | Per-round: clients responded, dropped, clean, poisoned, time |
| `client_round_details.csv` | Per-client per-round: aggregation inclusion, reputation weight |
| `poisoning_detection.csv` | Per-client per-round: z-scores, anomaly score, flagged status |
| `reputation_history.csv` | Per-client per-round: reputation value, validation result |
| `communication_metrics.csv` | Per-round: upload/broadcast bytes, centralized comparison |
| `participation_metrics.csv` | Per-round: sampled, responded, dropped, participation rate |
| `final_reputations.csv` | End-of-session reputation with tier info |

---

## Configuration

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `NUM_CLIENTS` | `server.py` | 10 | Number of FL clients (10–20 supported) |
| `NUM_ROUNDS` | `server.py` | 5 | Number of FL training rounds |
| `Z_THRESHOLD` | `server.py` | 1.5 | Poisoning detection sensitivity |
| `FRACTION_FIT` | `server.py` | 0.7 | Fraction of clients sampled per round |
| `ROUND_TIMEOUT` | `server.py` | 45s | Seconds to wait for clients per round |
| `SERVER_ADDRESS` | `server.py` | 0.0.0.0:9090 | Flower gRPC server address |
| `DEVICE_TIERS` | `client.py` | 3 tiers | Tier configs: epochs, batch, data fraction, dropout |
| `initial_reputation` | `FederatedLearning.sol` | 100 | Starting reputation score |

---

## Key Mathematical Formulas

| Formula | Expression | Source |
|---------|-----------|--------|
| Update Magnitude | `mag = (1/N) Σ|w_j|` | `poisoning_detector.py` |
| Update Direction | `d = (W_new − W_old) / ‖W_new − W_old‖₂` | `poisoning_detector.py` |
| Cosine Score | `cos = (1/(K−1)) Σ(d_i · d_j)` | `poisoning_detector.py` |
| Anomaly Score | `0.6 × z_mag + 0.4 × z_cos` | `poisoning_detector.py` |
| Poisoning Flag | `score > 1.5` | `poisoning_detector.py` |
| Reputation Weight | `w_i = rep_i / Σ rep_j` | `server.py` |
| Global Model | `W_global = Σ w_i × W_i` | `server.py` |
| BW Savings | `(1 − bytes_FL / bytes_centralized) × 100%` | `server.py` |
| Centralized Bytes | `148,517 × 41 × 4 = 24,356,788 bytes` | `server.py` |
| Feature Scaling | `x_scaled = (x − μ) / σ` | `data_loader.py` |
| Label Flip Attack | `y_poisoned = (y + 1) mod 5` | `attack_simulator.py` |
| Noise Attack | `X_poisoned = X + N(0, σ=2.0)` | `attack_simulator.py` |
| Gradient Scaling | `W_attack = W_global + 10 × (W_trained − W_global)` | `attack_simulator.py` |

---

## Acknowledgments

- [Flower](https://flower.dev/) — Federated Learning framework
- [Ganache](https://trufflesuite.com/ganache/) — Local Ethereum blockchain
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — Network intrusion detection dataset
- [TensorFlow](https://www.tensorflow.org/) — Machine learning framework
