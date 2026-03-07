# Secure Blockchain-Integrated Federated Learning for IoT Edge Devices

A privacy-preserving distributed machine learning system that combines **Federated Learning** with **Blockchain** technology to enable secure, verifiable model training across IoT edge devices. The system features anomaly detection using the NSL-KDD dataset, z-score-based poisoning detection, reputation-weighted aggregation, 3-tier device heterogeneity, and dropout/async handling — all recorded on an immutable blockchain ledger.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        UBUNTU SERVER                             │
│                                                                  │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│   │   GANACHE     │  │   TRUFFLE    │  │   FLOWER SERVER    │    │
│   │   Local ETH   │  │   Smart      │  │   FL Coordinator   │    │
│   │   Blockchain  │  │   Contract   │  │   SecureFedAvg     │    │
│   │   Port: 7545  │  │   Solidity   │  │   Port: 9090       │    │
│   └──────┬───────┘  └──────────────┘  └─────────┬──────────┘    │
│          │          Blockchain Verification       │ FL Protocol   │
└──────────┼──────────────────────────────────────┼───────────────┘
           │                                       │
     ┌─────┴─────┬──────────┬──────────┬──────────┴────┐
     ▼           ▼          ▼          ▼               ▼
┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌──────────────┐
│Client 0 ││Client 1 ││Client 2 ││Client 3 ││  Client 4..N │
│Tier 1   ││Tier 2   ││Tier 2   ││Tier 3   ││  Mixed Tiers │
│5 epochs ││3 epochs ││3 epochs ││1 epoch  ││  Configurable│
│100% data││70% data ││70% data ││40% data ││  Per-tier    │
└─────────┘└─────────┘└─────────┘└─────────┘└──────────────┘

Dataset: NSL-KDD (Network Intrusion Detection) | 41 Features | 5 Classes
Clients: 10-20 supported | Device Tiers: 3 | Dropout: Simulated
```

## Features

- **Federated Learning**: Distributed model training using Flower framework with FedAvg aggregation — raw data never leaves individual devices
- **Blockchain Integration**: Immutable audit trail of all model updates, validations, and global model hashes via Ethereum smart contracts
- **Poisoning Detection**: Z-score-based anomaly detection using weight magnitude (60%) and cosine similarity (40%) to identify malicious participants
- **Reputation System**: Blockchain-managed trust scores that increase for honest participants (+10) and decrease for flagged ones (-20), influencing aggregation weight
- **Reputation-Weighted Aggregation**: Clean clients contribute to the global model proportional to their reputation, reducing the influence of less trusted devices
- **3-Tier Device Heterogeneity**: Simulates real IoT deployments with Powerful Edge Servers (Tier 1), Mid-Range IoT Devices (Tier 2), and Constrained Sensor Nodes (Tier 3), each with different epochs, batch sizes, data fractions, and learning rates
- **Dropout/Async Handling**: Clients can simulate connection failures based on tier reliability; the server gracefully continues with available clients
- **Communication Metrics**: Per-round bandwidth tracking with centralized comparison showing 94-98% savings
- **Attack Simulation**: Built-in tools to simulate label flipping, noise injection, and scaling attacks for security testing
- **Research Data Pipeline**: Exports 9 CSV files per session covering round summaries, poisoning scores, reputation history, tier assignments, communication metrics, and participation data
- **Centralized Baseline**: Standalone training script for direct accuracy comparison against federated results
- **Automated Testing**: 34 unit tests covering model, data pipeline, poisoning detector, attack simulator, client utilities, and blockchain helper
- **NSL-KDD Dataset**: Automatic download and preprocessing with 5-class classification (Normal, DoS, Probe, R2L, U2R)

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Federated Learning | Flower (flwr) | FL coordination and FedAvg aggregation |
| Machine Learning | TensorFlow / Keras | Neural network model for anomaly detection |
| Blockchain | Ganache + Truffle | Local Ethereum blockchain simulation |
| Smart Contracts | Solidity | Device registration, model tracking, reputation |
| Blockchain Client | Web3.py | Python-to-blockchain interaction |
| Dataset | NSL-KDD | Network intrusion detection (41 features, 5 classes) |

## Device Tiers

| Tier | Device Type | Epochs | Batch Size | Data Fraction | Learning Rate | Dropout Prob |
|------|-------------|--------|------------|---------------|---------------|-------------|
| 1 | Powerful Edge Server | 5 | 16 | 100% | 0.001 | 0% |
| 2 | Mid-Range IoT Device | 3 | 32 | 70% | 0.001 | 10% |
| 3 | Constrained Sensor Node | 1 | 64 | 40% | 0.0005 | 25% |

Distribution: ~20% Tier 1, ~40% Tier 2, ~40% Tier 3 (repeating pattern: T1, T2, T2, T3, T3)

## Project Structure

```
fl-project/
├── contracts/
│   └── FederatedLearning.sol    # Solidity smart contract
├── migrations/
│   ├── 1_initial_migration.js
│   └── 2_deploy_contracts.js    # Contract deployment script
├── tests/
│   └── test_system.py           # 34 automated tests
├── results/                     # Research data output (CSV/JSON)
├── logs/                        # Client and server logs
├── build/                       # Compiled contract artifacts
├── data/                        # NSL-KDD dataset (auto-downloaded)
├── server.py                    # Flower server with SecureFedAvg strategy
├── client.py                    # Flower client with tier heterogeneity
├── attack_simulator.py          # Malicious client for testing
├── model.py                     # TensorFlow/Keras neural network
├── data_loader.py               # NSL-KDD download, preprocessing, partitioning
├── blockchain_helper.py         # Web3.py blockchain interaction wrapper
├── poisoning_detector.py        # Z-score anomaly detection
├── baseline.py                  # Centralized training baseline
├── run.sh                       # One-command full system launcher
├── stop.sh                      # Kill all running processes
├── launch_clients.sh            # Launch N clients in background
├── requirements.txt             # Python dependencies
├── truffle-config.js            # Truffle network configuration
└── README.md
```

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
# If you have sudo access:
sudo npm install -g truffle ganache

# Without sudo:
mkdir -p ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
npm install -g truffle ganache
```

## Usage

### Quick Start (One Command)

```bash
# 10 clients, 5 rounds, dropout enabled
./run.sh 10 5

# 10 clients, 5 rounds, dropout disabled
./run.sh 10 5 nodrop

# 20 clients, 3 rounds
./run.sh 20 3

# Stop everything
./stop.sh
```

### Manual Launch (3 Terminals)

**Terminal 1** — Ganache Blockchain:
```bash
ganache --port 7545 --accounts 25
```

**Terminal 2** — Deploy contracts and start FL Server:
```bash
source venv/bin/activate
truffle migrate --reset --network development
python server.py 10 5
```

**Terminal 3** — Launch clients:
```bash
./launch_clients.sh 10
```

### Running an Attack Simulation

Replace one or more honest clients with malicious ones:

```bash
# Label flip attack (shifts all training labels by 1)
python attack_simulator.py 0 label_flip

# Noise injection attack (adds Gaussian noise to training data)
python attack_simulator.py 0 noise_injection

# Scaling attack (amplifies weight updates 10x)
python attack_simulator.py 0 scaling
```

**Example attack scenario** — 1 attacker + 9 honest clients:

| Terminal | Command | Role |
|----------|---------|------|
| 1 | `ganache --port 7545 --accounts 25` | Blockchain |
| 2 | `python server.py 10 5` | FL Server |
| 3 | `python attack_simulator.py 0 label_flip` | Attacker |
| 4-11 | `python client.py 1` through `python client.py 9` | Honest |

### Running the Centralized Baseline

```bash
python baseline.py        # 15 epochs default
python baseline.py 20     # custom epochs
```

### Running Tests

```bash
python tests/test_system.py
```

## Smart Contract

The `FederatedLearning.sol` contract manages:

- **Device Registration**: Each IoT device registers with a unique ID and receives an initial reputation of 100
- **Model Update Tracking**: SHA-256 hashes of model weights are stored on-chain with timestamps and accuracy metrics
- **Validation**: The server (contract owner) validates each client's update based on poisoning detection results
- **Reputation Management**: Reputation increases by 10 for passed validations (max 200) and decreases by 20 for failures (min 10)
- **Global Model Recording**: Each round's aggregated global model hash is immutably stored

## Poisoning Detection

The system uses a dual-metric z-score approach:

1. **Weight Magnitude Analysis** (60% weight): Computes the average absolute value of each client's model weights and flags statistical outliers
2. **Cosine Similarity Analysis** (40% weight): Measures directional alignment between each client's update and all other clients — divergent directions indicate adversarial behavior

A combined anomaly score above the threshold (default: 1.5) flags the client as suspicious. Flagged clients are excluded from aggregation and their reputation drops on the blockchain.

## Results

### Centralized Baseline

| Metric | Value |
|--------|-------|
| Final Test Accuracy | 81.1% |
| Training Epochs | 15 |
| FL Target (within 5%) | 76.1% - 81.1% |

### Normal Run (10 Honest Clients)

| Round | Loss | Communication | Savings vs Centralized |
|-------|------|---------------|----------------------|
| 1 | 0.163 | 1,240 KB | 94.8% |
| 2 | 0.102 | 1,240 KB | 94.8% |
| 3 | 0.083 | 1,240 KB | 94.8% |
| 4 | 0.076 | 1,240 KB | 94.8% |
| 5 | 0.071 | 1,240 KB | 94.8% |

### Attack Run (1 Attacker + 9 Honest)

The poisoning detection system identifies malicious clients and reduces their influence through reputation-weighted aggregation. Honest clients reach 150 reputation while flagged attackers drop to 80-90.

### Dropout Run (10 Clients, Dropout Enabled)

| Metric | Value |
|--------|-------|
| Avg Participation Rate | 93.1% |
| Total Dropouts | 3 across 5 rounds |
| Model Performance | Comparable to no-dropout |

## Research Data Output

Each session exports 9 files to the `results/` directory:

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

## Configuration

Key parameters can be adjusted in the source files:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `NUM_CLIENTS` | `server.py` | 10 | Number of FL clients (10-20 supported) |
| `NUM_ROUNDS` | `server.py` | 5 | Number of FL training rounds |
| `Z_THRESHOLD` | `server.py` | 1.5 | Poisoning detection sensitivity |
| `FRACTION_FIT` | `server.py` | 1.0 | Fraction of clients sampled per round |
| `ROUND_TIMEOUT` | `server.py` | 120 | Seconds to wait for clients per round |
| `DEVICE_TIERS` | `client.py` | 3 tiers | Tier configs: epochs, batch, data fraction, dropout |
| `initial_reputation` | `FederatedLearning.sol` | 100 | Starting reputation score |

### CLI Arguments

```bash
# Server: python server.py [num_clients] [num_rounds]
python server.py 20 10

# Client: python client.py [client_id] [num_clients] [flags]
python client.py 0 20 --no-dropout --random-tiers

# Attack: python attack_simulator.py [client_id] [attack_type] [num_clients]
python attack_simulator.py 0 label_flip 20

# Baseline: python baseline.py [num_epochs]
python baseline.py 20

# Full system: ./run.sh [num_clients] [num_rounds] [nodrop]
./run.sh 20 10 nodrop
```

## Testing

34 automated tests across 6 components:

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

## License

This project is developed as part of academic research. See LICENSE for details.

## Acknowledgments

- [Flower](https://flower.dev/) — Federated Learning framework
- [Ganache](https://trufflesuite.com/ganache/) — Local Ethereum blockchain
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — Network intrusion detection dataset
- [TensorFlow](https://www.tensorflow.org/) — Machine learning framework
