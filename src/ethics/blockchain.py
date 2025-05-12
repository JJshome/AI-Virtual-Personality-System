"""
Blockchain verification module for the AI-based Virtual Personality System.

This module provides blockchain-based verification functionality for ensuring data integrity,
traceability, and security in the virtual personality system.
"""

import hashlib
import time
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class BlockchainVerifier:
    """
    Verify data integrity using blockchain technology.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the blockchain verifier.
        
        Args:
            config: Configuration dictionary for blockchain verification
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.blockchain_type = config.get('blockchain_type', 'simulated')
        self.blockchain = []  # Simulated blockchain for development purposes
        self.transaction_history = []
        
        # Blockchain connection parameters (would be used in a real implementation)
        self.api_key = config.get('api_key', '')
        self.api_endpoint = config.get('api_endpoint', '')
        self.network = config.get('network', 'testnet')
        
        logger.info(f"Initialized BlockchainVerifier with {self.blockchain_type} blockchain")
        
        # Create genesis block if using simulated blockchain
        if self.blockchain_type == 'simulated' and self.enabled:
            self._create_genesis_block()
    
    def _create_genesis_block(self) -> None:
        """
        Create the genesis block for the simulated blockchain.
        """
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'data': 'Genesis Block for AI Virtual Personality System',
            'previous_hash': '0' * 64,
            'nonce': 0
        }
        
        # Hash the block
        genesis_block['hash'] = self._hash_block(genesis_block)
        
        # Add to blockchain
        self.blockchain.append(genesis_block)
        
        logger.info(f"Created genesis block: {genesis_block['hash']}")
    
    def _hash_block(self, block: Dict[str, Any]) -> str:
        """
        Hash a block.
        
        Args:
            block: Block to hash
            
        Returns:
            Block hash
        """
        # Convert block to string and hash
        block_string = json.dumps(
            {k: v for k, v in block.items() if k != 'hash'},
            sort_keys=True
        )
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def _create_block(self, data: Any) -> Dict[str, Any]:
        """
        Create a new block.
        
        Args:
            data: Data to store in the block
            
        Returns:
            New block
        """
        previous_block = self.blockchain[-1]
        new_block = {
            'index': previous_block['index'] + 1,
            'timestamp': time.time(),
            'data': data,
            'previous_hash': previous_block['hash'],
            'nonce': 0
        }
        
        # Simple proof of work (in a real implementation, this would be more complex)
        while True:
            new_block['hash'] = self._hash_block(new_block)
            if new_block['hash'].startswith('0'):  # Simple difficulty
                break
            new_block['nonce'] += 1
        
        return new_block
    
    def verify(self, data: str) -> str:
        """
        Verify data on blockchain.
        
        Args:
            data: Data to verify
            
        Returns:
            Verification hash/ID
        """
        if not self.enabled:
            logger.info("Blockchain verification disabled")
            return ""
        
        # In a real implementation, this would interact with a blockchain
        # For simulation, we'll add to our simulated blockchain
        
        if self.blockchain_type == 'simulated':
            # Create a new block
            block = self._create_block(data)
            
            # Add to blockchain
            self.blockchain.append(block)
            
            logger.info(f"Data verified on blockchain with hash: {block['hash']}")
            
            # Record transaction
            transaction = {
                'timestamp': time.time(),
                'data_hash': hashlib.sha256(data.encode()).hexdigest(),
                'block_hash': block['hash'],
                'block_index': block['index']
            }
            self.transaction_history.append(transaction)
            
            return block['hash']
        else:
            # This would be replaced with actual blockchain interaction in a real system
            # For example, calling a blockchain API
            
            verification_hash = hashlib.sha256(data.encode()).hexdigest()
            
            logger.info(f"Data verified on blockchain with hash: {verification_hash}")
            
            return verification_hash
    
    def verify_integrity(self, data: str, verification_hash: str) -> bool:
        """
        Verify the integrity of data using a verification hash.
        
        Args:
            data: Data to verify
            verification_hash: Verification hash to check against
            
        Returns:
            True if data integrity is verified, False otherwise
        """
        if not self.enabled:
            logger.info("Blockchain verification disabled")
            return True
        
        if self.blockchain_type == 'simulated':
            # Check if the hash exists in our blockchain
            for block in self.blockchain:
                if block['hash'] == verification_hash:
                    # Verify the data matches
                    if isinstance(block['data'], str) and block['data'] == data:
                        logger.info("Data integrity verified")
                        return True
                    else:
                        # For non-string data, we'd need to hash it first
                        data_hash = hashlib.sha256(data.encode()).hexdigest()
                        for transaction in self.transaction_history:
                            if transaction['block_hash'] == verification_hash and transaction['data_hash'] == data_hash:
                                logger.info("Data integrity verified")
                                return True
            
            logger.warning("Data integrity verification failed")
            return False
        else:
            # This would be replaced with actual blockchain verification in a real system
            
            computed_hash = hashlib.sha256(data.encode()).hexdigest()
            
            is_verified = computed_hash == verification_hash
            
            if is_verified:
                logger.info("Data integrity verified")
            else:
                logger.warning("Data integrity verification failed")
            
            return is_verified
    
    def verify_chain(self) -> bool:
        """
        Verify the integrity of the entire blockchain.
        
        Returns:
            True if blockchain integrity is verified, False otherwise
        """
        if not self.enabled or self.blockchain_type != 'simulated':
            logger.info("Chain verification not applicable")
            return True
        
        for i in range(1, len(self.blockchain)):
            current_block = self.blockchain[i]
            previous_block = self.blockchain[i - 1]
            
            # Check hash integrity
            if current_block['hash'] != self._hash_block(current_block):
                logger.warning(f"Block {i} hash is invalid")
                return False
            
            # Check chain integrity
            if current_block['previous_hash'] != previous_block['hash']:
                logger.warning(f"Block {i} previous hash does not match block {i-1} hash")
                return False
        
        logger.info("Blockchain integrity verified")
        return True
    
    def get_transaction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get transaction history.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction records
        """
        if not self.enabled:
            logger.info("Blockchain verification disabled")
            return []
        
        if self.blockchain_type == 'simulated':
            # Return most recent transactions first
            return sorted(
                self.transaction_history,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:limit]
        else:
            # This would be replaced with actual blockchain query in a real system
            logger.info("Transaction history not available for non-simulated blockchain")
            return []


# Create a factory function to make it easy to create blockchain verifiers
def create_blockchain_verifier(config: Dict[str, Any] = None) -> BlockchainVerifier:
    """
    Create a blockchain verifier.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        BlockchainVerifier instance
    """
    if config is None:
        config = {'enabled': False, 'blockchain_type': 'simulated'}
    
    return BlockchainVerifier(config)
