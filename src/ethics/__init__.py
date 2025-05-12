"""
Ethics and Security Management package for the AI-based Virtual Personality System.

This package provides components for ensuring that virtual personalities operate ethically
and securely, protecting user privacy, preventing misuse, and adhering to ethical guidelines.
"""

from .manager import EthicsSecurityManager, create_default_guidelines
from .content_filter import ContentFilter, create_content_filter
from .blockchain import BlockchainVerifier, create_blockchain_verifier
from .consent import ConsentManager, create_consent_manager

__all__ = [
    'EthicsSecurityManager',
    'ContentFilter',
    'BlockchainVerifier',
    'ConsentManager',
    'create_default_guidelines',
    'create_content_filter',
    'create_blockchain_verifier',
    'create_consent_manager'
]
