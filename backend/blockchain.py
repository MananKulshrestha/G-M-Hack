import json
import os
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from dotenv import load_dotenv, find_dotenv

# Load env vars
load_dotenv(find_dotenv())

class DigitalNotary:
    def __init__(self):
        # 1. Connect to RPC
        self.rpc_url = os.getenv("RPC_URL", "https://rpc-amoy.polygon.technology/")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        # --- CRITICAL: Inject Middleware for Polygon Amoy ---
        # This handles the extra data in block headers that causes standard web3 to crash
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to RPC: {self.rpc_url}")

        # 2. Setup Account
        self.private_key = os.getenv("PRIVATE_KEY")
        if not self.private_key:
            print("⚠️ PRIVATE_KEY not found. Blockchain features will be disabled.")
            raise ValueError("Missing Private Key")
        
        self.account = self.w3.eth.account.from_key(self.private_key)
        self.address = self.account.address

        # 3. Setup Contract
        self.contract_address = os.getenv("CONTRACT_ADDRESS")
        if not self.contract_address:
             print("⚠️ CONTRACT_ADDRESS not found.")
             raise ValueError("Missing Contract Address")

        # Resolve ABI path relative to this script's location
        abi_path = os.path.join(os.path.dirname(__file__), 'contracts', 'contract_abi.json')
        
        if not os.path.exists(abi_path):
            raise FileNotFoundError(f"ABI file not found at {abi_path}")

        with open(abi_path, 'r') as f:
            self.contract_abi = json.load(f)
            
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)

    def register_verification(self, file_hash: str, status: str) -> str:
        """
        Writes the file hash and status to the blockchain using EIP-1559.
        Returns the Transaction Hash.
        """
        try:
            # 1. Get the Nonce
            nonce = self.w3.eth.get_transaction_count(self.address, 'pending')

            # 2. Prepare Transaction Data
            tx_func = self.contract.functions.registerMedia(file_hash, status)
            
            # 3. Fee Estimation (EIP-1559)
            # Fetch base fee from latest block
            latest_block = self.w3.eth.get_block("latest")
            base_fee = latest_block['baseFeePerGas']
            
            # Set priority fee (tip) - Amoy usually requires 25-30 gwei minimum to be fast
            priority_fee = self.w3.to_wei(35, 'gwei')
            
            # Max fee = (2 * base_fee) + priority_fee
            max_fee = (base_fee * 2) + priority_fee

            # 4. Build Transaction
            tx_data = tx_func.build_transaction({
                'chainId': 80002,  # Polygon Amoy Chain ID
                'from': self.address,
                'nonce': nonce,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': priority_fee,
                'gas': 300000  # Gas limit buffer
            })

            # 5. Sign the Transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx_data, self.private_key)

            # 6. Broadcast to Network
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            return self.w3.to_hex(tx_hash)

        except Exception as e:
            print(f"Blockchain Error: {e}")
            return None