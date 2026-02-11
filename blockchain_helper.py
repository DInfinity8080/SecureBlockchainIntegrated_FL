import json
import hashlib
from web3 import Web3

class BlockchainHelper:
    def __init__(self, ganache_url='http://127.0.0.1:7545', contract_address=None):
        self.w3 = Web3(Web3.HTTPProvider(ganache_url))
        
        if not self.w3.is_connected():
            raise ConnectionError("Cannot connect to Ganache")
        
        print(f"Connected to blockchain: {ganache_url}")
        
        with open('build/contracts/FederatedLearning.json', 'r') as f:
            contract_json = json.load(f)
            self.abi = contract_json['abi']
        
        if contract_address:
            self.contract_address = Web3.to_checksum_address(contract_address)
        else:
            networks = contract_json.get('networks', {})
            if networks:
                network_id = list(networks.keys())[-1]
                self.contract_address = Web3.to_checksum_address(
                    networks[network_id]['address']
                )
            else:
                raise ValueError("No contract address found")
        
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=self.abi
        )
        
        self.accounts = self.w3.eth.accounts
        print(f"Contract at: {self.contract_address}")
        print(f"Available accounts: {len(self.accounts)}")
    
    def register_device(self, device_id, account_index=0):
        account = self.accounts[account_index]
        tx = self.contract.functions.registerDevice(device_id).transact({
            'from': account,
            'gas': 500000
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        print(f"Device '{device_id}' registered | TX: {receipt.transactionHash.hex()}")
        return receipt
    
    def submit_model_update(self, model_weights, accuracy, account_index=0):
        model_hash = self._hash_weights(model_weights)
        account = self.accounts[account_index]
        accuracy_int = int(accuracy * 100)
        
        tx = self.contract.functions.submitModelUpdate(
            model_hash, accuracy_int
        ).transact({
            'from': account,
            'gas': 500000
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        print(f"Model update submitted | Hash: {model_hash[:16]}... | Accuracy: {accuracy:.4f}")
        return receipt, model_hash
    
    def validate_update(self, device_address, round_num, passed, owner_index=0):
        owner = self.accounts[owner_index]
        tx = self.contract.functions.validateUpdate(
            Web3.to_checksum_address(device_address),
            round_num,
            passed
        ).transact({
            'from': owner,
            'gas': 500000
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        print(f"Validation {'PASSED' if passed else 'FAILED'} for {device_address[:10]}...")
        return receipt
    
    def update_global_model(self, model_weights, owner_index=0):
        model_hash = self._hash_weights(model_weights)
        owner = self.accounts[owner_index]
        
        tx = self.contract.functions.updateGlobalModel(model_hash).transact({
            'from': owner,
            'gas': 500000
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        current_round = self.contract.functions.currentRound().call()
        print(f"Global model updated | Round: {current_round} | Hash: {model_hash[:16]}...")
        return receipt
    
    def get_reputation(self, account_index=0):
        account = self.accounts[account_index]
        return self.contract.functions.getDeviceReputation(account).call()
    
    def get_device_count(self):
        return self.contract.functions.deviceCount().call()
    
    def _hash_weights(self, weights):
        concat = b''
        for w in weights:
            concat += w.tobytes()
        return hashlib.sha256(concat).hexdigest()

if __name__ == '__main__':
    bc = BlockchainHelper()
    print(f"Current round: {bc.contract.functions.currentRound().call()}")
    print(f"Registered devices: {bc.get_device_count()}")
