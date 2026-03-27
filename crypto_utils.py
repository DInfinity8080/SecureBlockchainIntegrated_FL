"""
Cryptographic utilities for Secure Federated Learning.
Provides ECDSA signing/verification and AES-GCM encryption/decryption
for gradient updates.
"""

import hashlib
import os
import numpy as np
from typing import List, Tuple, Optional

from eth_account import Account
from eth_account.messages import encode_defunct
from web3 import Web3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CryptoManager:
    """Handles ECDSA signing and AES-GCM encryption for FL weight updates."""

    def __init__(self, private_key_hex: str):
        """
        Initialize with an Ethereum private key (hex string, with or without 0x prefix).
        """
        if not private_key_hex.startswith('0x'):
            private_key_hex = '0x' + private_key_hex
        self.private_key = private_key_hex
        self.account = Account.from_key(self.private_key)
        self.address = self.account.address

    def get_address(self) -> str:
        return self.address

    # ── Hashing ───────────────────────────────────────────────────

    @staticmethod
    def hash_weights(weights: List[np.ndarray]) -> str:
        """Compute SHA-256 hash of concatenated weight bytes."""
        concat = b''
        for w in weights:
            concat += w.tobytes()
        return hashlib.sha256(concat).hexdigest()

    # ── ECDSA Signing / Verification ──────────────────────────────

    def sign_weights(self, weights: List[np.ndarray]) -> Tuple[str, str]:
        """
        Sign the SHA-256 hash of model weights with the Ethereum private key.
        Returns (hash_hex, signature_hex).
        """
        weight_hash = self.hash_weights(weights)
        message = encode_defunct(text=weight_hash)
        signed = Account.sign_message(message, private_key=self.private_key)
        signature_hex = signed.signature.hex()
        return weight_hash, signature_hex

    @staticmethod
    def verify_signature(hash_hex: str, signature_hex: str, expected_address: str) -> bool:
        """
        Verify that the signature over hash_hex was produced by expected_address.
        """
        try:
            message = encode_defunct(text=hash_hex)
            if not signature_hex.startswith('0x'):
                signature_hex = '0x' + signature_hex
            recovered = Account.recover_message(message, signature=signature_hex)
            return recovered.lower() == expected_address.lower()
        except Exception:
            return False

    # ── AES-GCM Encryption / Decryption ──────────────────────────

    @staticmethod
    def generate_aes_key() -> bytes:
        """Generate a random 256-bit AES key."""
        return AESGCM.generate_key(bit_length=256)

    @staticmethod
    def encrypt_weights(weights: List[np.ndarray], aes_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encrypt serialized model weights using AES-256-GCM.
        Returns (nonce, ciphertext).
        """
        plaintext = b''
        for w in weights:
            plaintext += w.tobytes()
        nonce = os.urandom(12)
        aesgcm = AESGCM(aes_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        return nonce, ciphertext

    @staticmethod
    def decrypt_weights(nonce: bytes, ciphertext: bytes, aes_key: bytes,
                        weight_shapes: List[Tuple], weight_dtypes: List[np.dtype]) -> List[np.ndarray]:
        """
        Decrypt AES-256-GCM ciphertext back into a list of numpy weight arrays.
        Requires the original shapes and dtypes to reconstruct arrays.
        """
        aesgcm = AESGCM(aes_key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        weights = []
        offset = 0
        for shape, dtype in zip(weight_shapes, weight_dtypes):
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            arr = np.frombuffer(plaintext[offset:offset + size], dtype=dtype).reshape(shape)
            weights.append(arr.copy())
            offset += size
        return weights

    @staticmethod
    def get_weight_metadata(weights: List[np.ndarray]) -> Tuple[List[Tuple], List[str]]:
        """Extract shapes and dtype strings from weights for reconstruction after decryption."""
        shapes = [w.shape for w in weights]
        dtypes = [str(w.dtype) for w in weights]
        return shapes, dtypes


def get_ganache_private_keys(ganache_url: str = 'http://127.0.0.1:7545') -> dict:
    """
    Retrieve private keys from Ganache using the eth_accounts + personal namespace,
    or by parsing the Ganache log file.
    Returns {account_index: private_key_hex}.
    """
    # Try parsing the Ganache log first (most reliable)
    keys = _parse_ganache_log()
    if keys:
        return keys

    # Fallback: try the Ganache RPC method (works with ganache v7+)
    try:
        w3 = Web3(Web3.HTTPProvider(ganache_url))
        accounts = w3.eth.accounts
        keys = {}
        # Ganache exposes private keys via evm_dumpState or they can be
        # retrieved if started with --wallet.deterministic
        # Try fetching via provider request
        response = w3.provider.make_request("evm_getAccountNonce", [accounts[0]])
        # If we got here, try personal namespace
        for i, account in enumerate(accounts):
            try:
                resp = w3.provider.make_request("eth_sign", [account, "0x"])
                if resp and 'result' in resp:
                    keys[i] = resp['result']
            except Exception:
                pass
        if keys:
            return keys
    except Exception:
        pass

    return {}


def _parse_ganache_log(log_path: str = 'logs/ganache.log') -> dict:
    """Parse Ganache startup log to extract private keys."""
    keys = {}
    try:
        with open(log_path, 'r') as f:
            content = f.read()

        in_private_keys = False
        for line in content.split('\n'):
            line = line.strip()
            if 'Private Keys' in line or 'private keys' in line.lower():
                in_private_keys = True
                continue
            if in_private_keys and line.startswith('('):
                # Format: (0) 0xabcdef...
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    idx_str = parts[0].strip('()')
                    key_hex = parts[1].strip()
                    try:
                        idx = int(idx_str)
                        if key_hex.startswith('0x') and len(key_hex) == 66:
                            keys[idx] = key_hex
                    except ValueError:
                        pass
            elif in_private_keys and line and not line.startswith('(') and not line.startswith('='):
                # End of private keys section
                if keys:
                    break
    except FileNotFoundError:
        pass
    return keys
