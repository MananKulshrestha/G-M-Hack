import json
import os
import time
from web3 import Web3
from dotenv import load_dotenv, find_dotenv

# Automatically find the .env file (searches parent directories) and load it
load_dotenv(find_dotenv())

class DigitalNotary:
    def __init__(self):
        # 1. Connect to Polygon Amoy
        self.rpc_url = os.getenv("RPC_URL", "https://rpc-amoy.polygon.technology/")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to RPC: {self.rpc_url}")

        # 2. Setup Account
        self.private_key = os.getenv("PRIVATE_KEY")
        if not self.private_key:
            raise ValueError("PRIVATE_KEY not found in environment variables. Ensure .env is correctly located.")
        
        self.account = self.w3.eth.account.from_key(self.private_key)
        self.address = self.account.address

        # 3. Setup Contract
        self.contract_address = os.getenv("CONTRACT_ADDRESS")
        if not self.contract_address:
             raise ValueError("CONTRACT_ADDRESS not found in environment variables.")

        # Resolve ABI path relative to this script's location
        abi_path = os.path.join(os.path.dirname(__file__), 'contracts', 'contract_abi.json')
        
        with open(abi_path, 'r') as f:
            self.contract_abi = json.load(f)
            
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)

    def register_verification(self, file_hash: str, status: str) -> str:
        """
        Writes the file hash and status to the blockchain.
        Returns the Transaction Hash.
        """
        try:
            # 1. Get the Nonce (Prevent replay attacks/collisions)
            # 'pending' includes transactions in mempool to avoid "nonce too low" errors
            nonce = self.w3.eth.get_transaction_count(self.address, 'pending')

            # 2. Build the Transaction
            tx_data = self.contract.functions.registerMedia(
                file_hash, 
                status
            ).build_transaction({
                'chainId': 80002,  # Polygon Amoy Chain ID
                'gas': 300000,     # Slight buffer
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
                'from': self.address
            })

            # 3. Sign the Transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx_data, self.private_key)

            # 4. Broadcast to Network
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            # Return the hex string of the transaction hash
            return self.w3.to_hex(tx_hash)

        except Exception as e:
            print(f"Blockchain Error: {e}")
            return None