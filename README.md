# Secure Blockchain-Integrated Federated Learning for IoT Edge Devices

A privacy-preserving distributed machine learning system that combines **Federated Learning** with **Blockchain** technology to enable secure, verifiable model training across IoT edge devices. The system features anomaly detection using the NSL-KDD dataset, z-score-based poisoning detection, and reputation-weighted aggregation — all recorded on an immutable blockchain ledger.

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
│Client 0 ││Client 1 ││Client 2 ││Client 3 ││  Client 4    │
│TF Model ││TF Model ││TF Model ││TF Model ││  TF Model    │
│Flower   ││Flower   ││Flower   ││Flower   ││  Flower      │
│Web3.py  ││Web3.py  ││Web3.py  ││Web3.py  ││  Web3.py     │
│Partition││Partition ││Partition││Partition ││  Partition   │
│   A     ││   B     ││   C     ││   D     ││     E        │
└─────────┘└─────────┘└─────────┘└─────────┘└──────────────┘

Dataset: NSL-KDD (Network Intrusion Detection) | 41 Features | 5 Classes
```

## Features

- **Federated Learning**: Distributed model training using Flower framework with FedAvg aggregation — raw data never leaves individual devices
- **Blockchain Integration**: Immutable audit trail of all model updates, validations, and global model hashes via Ethereum smart contracts
- **Poisoning Detection**: Z-score-based anomaly detection using weight magnitude and cosine similarity analysis to identify malicious participants
- **Reputation System**: Blockchain-managed trust scores that increase for honest participants (+10) and decrease for flagged ones (-20), influencing aggregation weight
- **Reputation-Weighted Aggregation**: Clean clients contribute to the global model proportional to their reputation, reducing the influence of less trusted devices
- **Attack Simulation**: Built-in tools to simulate label flipping, noise injection, and scaling attacks for security testing
- **NSL-KDD Dataset**: Automatic download and preprocessing of the NSL-KDD network intrusion detection dataset with 5-class classification (Normal, DoS, Probe, R2L, U2R)

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Federated Learning | Flower (flwr) | FL coordination and FedAvg aggregation |
| Machine Learning | TensorFlow / Keras | Neural network model for anomaly detection |
| Blockchain | Ganache + Truffle | Local Ethereum blockchain simulation |
| Smart Contracts | Solidity | Device registration, model tracking, reputation |
| Blockchain Client | Web3.py | Python-to-blockchain interaction |
| Dataset | NSL-KDD | Network intrusion detection (41 features, 5 classes) |

## Project Structure

```
fl-project/
├── contracts/
│   └── FederatedLearning.sol    # Solidity smart contract
├── migrations/
│   ├── 1_initial_migration.js
│   └── 2_deploy_contracts.js    # Contract deployment script
├── build/                       # Compiled contract artifacts
├── data/                        # NSL-KDD dataset (auto-downloaded)
├── server.py                    # Flower server with SecureFedAvg strategy
├── client.py                    # Flower client for honest participants
├── attack_simulator.py          # Malicious client for testing
├── model.py                     # TensorFlow/Keras neural network
├── data_loader.py               # NSL-KDD download, preprocessing, partitioning
├── blockchain_helper.py         # Web3.py blockchain interaction wrapper
├── poisoning_detector.py        # Z-score anomaly detection
├── truffle-config.js            # Truffle network configuration
├── package.json
└── README.md
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm 9+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kaps-Jadeja/SecureBlockchainIntegrated_FL.git
cd SecureBlockchainIntegrated_FL
```

### 2. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow flwr web3 numpy pandas scikit-learn matplotlib
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

### 4. Initialize and Deploy Smart Contracts

```bash
# Start Ganache in a separate terminal
ganache --port 7545

# Compile and deploy
truffle compile
truffle migrate --network development
```

## Usage

### Running a Normal Federated Learning Session

You need **6 terminals** total:

**Terminal 1** — Ganache Blockchain:
```bash
ganache --port 7545
```

**Terminal 2** — Deploy contracts and start FL Server:
```bash
source venv/bin/activate
truffle migrate --reset --network development
python server.py
```

**Terminals 3–7** — Start 5 FL Clients:
```bash
source venv/bin/activate
python client.py 0    # Client 0 (Partition A)
python client.py 1    # Client 1 (Partition B)
python client.py 2    # Client 2 (Partition C)
python client.py 3    # Client 3 (Partition D)
python client.py 4    # Client 4 (Partition E)
```

The server waits for all 5 clients to connect, then begins federated training for 5 rounds.

### Running an Attack Simulation

Replace one or more honest clients with malicious ones:

```bash
# Label flip attack (randomizes training labels)
python attack_simulator.py 0 label_flip

# Noise injection attack (adds Gaussian noise to training data)
python attack_simulator.py 0 noise_injection

# Scaling attack (amplifies weight updates 10x)
python attack_simulator.py 0 scaling
```

**Example attack scenario** — 1 attacker + 4 honest clients:

| Terminal | Command | Role |
|----------|---------|------|
| 1 | `ganache --port 7545` | Blockchain |
| 2 | `python server.py` | FL Server |
| 3 | `python attack_simulator.py 0 label_flip` | Attacker |
| 4 | `python client.py 1` | Honest |
| 5 | `python client.py 2` | Honest |
| 6 | `python client.py 3` | Honest |
| 7 | `python client.py 4` | Honest |

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

### Normal Run (5 Honest Clients)

| Round | Loss | Reputations |
|-------|------|-------------|
| 1 | 0.109 | All: 110 |
| 2 | 0.094 | All: 120 |
| 3 | 0.096 | All: 130 |
| 4 | 0.050 | All: 140 |
| 5 | 0.045 | All: 150 |

### Attack Run (1 Attacker + 4 Honest)

| Round | Loss | Detection | Attacker Rep | Honest Rep |
|-------|------|-----------|--------------|------------|
| 1 | 0.109 | All clean | 110 | 110 |
| 2 | 0.094 | All clean | 120 | 120 |
| 3 | 0.096 | All clean | 130 | 130 |
| 4 | 0.050 | **Attacker flagged** | **110** | 140 |
| 5 | 0.045 | **Attacker flagged** | **90** | 150 |

The attacker's reputation dropped from 130 to 90 while honest clients reached 150. The attacker was excluded from aggregation in Rounds 4 and 5, resulting in improved model quality.

## Configuration

Key parameters can be adjusted in the source files:

| Parameter | File | Default | Description |
|-----------|------|---------|-------------|
| `num_rounds` | `server.py` | 5 | Number of FL training rounds |
| `z_threshold` | `server.py` | 1.5 | Poisoning detection sensitivity |
| `num_clients` | `client.py` | 5 | Number of data partitions |
| `epochs` | `client.py` | 3 | Local training epochs per round |
| `batch_size` | `client.py` | 32 | Training batch size |
| `initial_reputation` | `FederatedLearning.sol` | 100 | Starting reputation score |

## License

This project is developed as part of academic research. See LICENSE for details.

## Acknowledgments

- [Flower](https://flower.dev/) — Federated Learning framework
- [Ganache](https://trufflesuite.com/ganache/) — Local Ethereum blockchain
- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — Network intrusion detection dataset
- [TensorFlow](https://www.tensorflow.org/) — Machine learning framework
