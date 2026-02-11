import flwr as fl
import numpy as np
from model import create_model
from blockchain_helper import BlockchainHelper

class BlockchainFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, blockchain, **kwargs):
        super().__init__(**kwargs)
        self.blockchain = blockchain
        self.round_num = 0

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)

            try:
                self.blockchain.update_global_model(weights, owner_index=0)
                print(f"[Round {server_round}] Global model recorded on blockchain")
            except Exception as e:
                print(f"[Round {server_round}] Blockchain error: {e}")

            self.round_num = server_round

        return aggregated_parameters, aggregated_metrics

def start_server(num_rounds=5):
    model = create_model()
    initial_weights = model.get_weights()
    initial_parameters = fl.common.ndarrays_to_parameters(initial_weights)

    try:
        blockchain = BlockchainHelper()
        print("Blockchain connected for FL server")
    except Exception as e:
        print(f"Blockchain connection failed: {e}")
        blockchain = None

    if blockchain:
        strategy = BlockchainFedAvg(
            blockchain=blockchain,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            initial_parameters=initial_parameters,
        )
    else:
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            initial_parameters=initial_parameters,
        )

    fl.server.start_server(
        server_address="0.0.0.0:9090",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == '__main__':
    start_server(num_rounds=5)

