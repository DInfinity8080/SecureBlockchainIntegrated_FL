import json
import hashlib
import threading
from web3 import Web3

# Maximum seconds to wait for a single transaction receipt before giving up.
TX_TIMEOUT_S = 30

def _call_with_timeout(fn, timeout=TX_TIMEOUT_S):
    """Run *fn* in a thread; raise TimeoutError if it exceeds *timeout* seconds."""
    result = [None]
    exc    = [None]

    def target():
        try:
            result[0] = fn()
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(
            f"Blockchain call exceeded {timeout}s timeout — Ganache may be overloaded."
        )
    if exc[0] is not None:
        raise exc[0]
    return result[0]


class BlockchainHelper:
    def __init__(self, ganache_url='http://127.0.0.1:7545', contract_address=None):
        self.w3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': TX_TIMEOUT_S}))

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
        self.registered = set()
        print(f"Contract at: {self.contract_address}")
        print(f"Available accounts: {len(self.accounts)}")

    def _transact_and_wait(self, fn, from_account):
        """Send a transaction and block until its receipt arrives (with timeout)."""
        def _inner():
            tx = fn().transact({'from': from_account, 'gas': 500000})
            return self.w3.eth.wait_for_transaction_receipt(tx, timeout=TX_TIMEOUT_S)
        return _call_with_timeout(_inner, timeout=TX_TIMEOUT_S + 5)

    def ensure_registered(self, account_index, device_id=None):
        if account_index in self.registered:
            return
        account = self.accounts[account_index]
        try:
            is_reg = self.contract.functions.isDeviceRegistered(account).call()
            if not is_reg:
                if device_id is None:
                    device_id = f"device_{account_index}"
                self._transact_and_wait(
                    lambda: self.contract.functions.registerDevice(device_id),
                    account
                )
                print(f"Device '{device_id}' registered | Account: {account[:10]}...")
            self.registered.add(account_index)
        except TimeoutError as e:
            print(f"Registration timeout (skipping): {e}")
        except Exception as e:
            print(f"Registration check error: {e}")

    def register_device(self, device_id, account_index=0):
        self.ensure_registered(account_index, device_id)

    def submit_model_update(self, model_weights, accuracy, account_index=0):
        self.ensure_registered(account_index)
        model_hash = self._hash_weights(model_weights)
        account = self.accounts[account_index]
        accuracy_int = int(accuracy * 100)

        try:
            receipt = self._transact_and_wait(
                lambda: self.contract.functions.submitModelUpdate(model_hash, accuracy_int),
                account
            )
            print(f"Model update submitted | Hash: {model_hash[:16]}... | Accuracy: {accuracy:.4f}")
            return receipt, model_hash
        except TimeoutError as e:
            print(f"submit_model_update timeout (skipping): {e}")
            return None, model_hash

    def validate_update(self, device_address, round_num, passed, owner_index=0):
        owner = self.accounts[owner_index]
        try:
            receipt = self._transact_and_wait(
                lambda: self.contract.functions.validateUpdate(
                    Web3.to_checksum_address(device_address), round_num, passed
                ),
                owner
            )
            print(f"Validation {'PASSED' if passed else 'FAILED'} for {device_address[:10]}...")
            return receipt
        except TimeoutError as e:
            print(f"validate_update timeout (skipping): {e}")
            return None

    def update_global_model(self, model_weights, owner_index=0):
        model_hash = self._hash_weights(model_weights)
        owner = self.accounts[owner_index]
        try:
            receipt = self._transact_and_wait(
                lambda: self.contract.functions.updateGlobalModel(model_hash),
                owner
            )
            current_round = self.contract.functions.currentRound().call()
            print(f"Global model updated | Round: {current_round} | Hash: {model_hash[:16]}...")
            return receipt
        except TimeoutError as e:
            print(f"update_global_model timeout (skipping): {e}")
            return None

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
