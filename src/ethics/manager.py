"""
Ethics and Security Management Module for the AI-based Virtual Personality System.
This module ensures that virtual personalities operate ethically and securely,
protecting user privacy, preventing misuse, and adhering to ethical guidelines.
"""

import os
import logging
import time
import json
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class EthicsSecurityManager:
    """
    Manages ethics and security aspects of the virtual personality system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ethics and security manager.
        
        Args:
            config: Configuration dictionary for ethics and security management
        """
        self.config = config
        self.content_filtering = config.get('content_filtering', True)
        self.privacy_protection = config.get('privacy_protection', True)
        self.blockchain_verification = config.get('blockchain_verification', False)
        self.user_consent_required = config.get('user_consent_required', True)
        
        # Load ethical guidelines
        self.guidelines_path = config.get('ethical_guidelines_path', 'ethics/guidelines.yaml')
        self.guidelines = self._load_guidelines()
        
        # Initialize user consent registry
        self.user_consent = {}
        
        # Initialize content filters
        self.content_filters = self._initialize_filters()
        
        # Initialize security measures
        self.data_encryption = config.get('data_encryption', True)
        self.encryption_key = None
        if self.data_encryption:
            self.encryption_key = self._initialize_encryption()
        
        logger.info(f"Initialized EthicsSecurityManager with content filtering: {self.content_filtering}")
        logger.info(f"Privacy protection: {self.privacy_protection}, User consent required: {self.user_consent_required}")
        logger.info(f"Loaded {len(self.guidelines)} ethical guidelines")
    
    def _load_guidelines(self) -> Dict[str, Any]:
        """
        Load ethical guidelines from the specified path.
        
        Returns:
            Guidelines dictionary
        """
        guidelines = {
            'general': [
                "Respect user privacy and confidentiality.",
                "Be transparent about being an AI system.",
                "Do not pretend to have human experiences or emotions.",
                "Do not engage in harmful, illegal, or unethical activities.",
                "Do not discriminate based on personal characteristics.",
                "Provide accurate and balanced information.",
                "Acknowledge limitations and uncertainties.",
                "Prioritize user wellbeing and safety."
            ],
            'content': [
                "Avoid generating harmful, offensive, or misleading content.",
                "Do not create content that promotes illegal activities.",
                "Do not generate content that exploits or harms vulnerable groups.",
                "Respect copyright and intellectual property.",
                "Be mindful of cultural sensitivities and differences."
            ],
            'interaction': [
                "Be respectful and courteous in all interactions.",
                "Avoid manipulative or deceptive behavior.",
                "Do not engage in harassment or bullying.",
                "Respect user autonomy and choices.",
                "Be mindful of power dynamics in interactions."
            ],
            'domain_specific': {
                'entertainment': [
                    "Provide age-appropriate content.",
                    "Respect artistic integrity and creative expression.",
                    "Acknowledge sources and inspirations."
                ],
                'education': [
                    "Provide accurate and balanced educational information.",
                    "Respect different learning styles and abilities.",
                    "Avoid imposing personal biases in educational content."
                ],
                'healthcare': [
                    "Do not provide specific medical diagnoses or treatments.",
                    "Encourage consulting qualified healthcare professionals.",
                    "Respect medical privacy and confidentiality.",
                    "Be sensitive to health-related concerns and anxieties."
                ],
                'customer_service': [
                    "Provide fair and accurate information about products and services.",
                    "Respect customer rights and privacy.",
                    "Be transparent about limitations and capabilities."
                ],
                'financial': [
                    "Do not provide specific financial or investment advice.",
                    "Encourage consulting qualified financial professionals.",
                    "Be transparent about limitations in financial knowledge.",
                    "Respect financial privacy and confidentiality."
                ],
                'tourism': [
                    "Provide culturally sensitive travel information.",
                    "Respect local customs, traditions, and regulations.",
                    "Promote responsible and sustainable tourism."
                ]
            }
        }
        
        # If guidelines file exists, load it
        if os.path.exists(self.guidelines_path):
            try:
                with open(self.guidelines_path, 'r') as f:
                    file_guidelines = yaml.safe_load(f)
                
                # Merge with default guidelines
                if file_guidelines:
                    for category, items in file_guidelines.items():
                        if category == 'domain_specific':
                            for domain, domain_items in items.items():
                                if domain in guidelines['domain_specific']:
                                    guidelines['domain_specific'][domain].extend(domain_items)
                                else:
                                    guidelines['domain_specific'][domain] = domain_items
                        elif category in guidelines:
                            guidelines[category].extend(items)
                        else:
                            guidelines[category] = items
            except Exception as e:
                logger.error(f"Failed to load guidelines from {self.guidelines_path}: {e}")
                logger.info("Using default guidelines")
        else:
            logger.warning(f"Guidelines file not found at {self.guidelines_path}")
            logger.info("Using default guidelines")
            
            # Save default guidelines for reference
            try:
                os.makedirs(os.path.dirname(self.guidelines_path), exist_ok=True)
                with open(self.guidelines_path, 'w') as f:
                    yaml.dump(guidelines, f, default_flow_style=False)
                logger.info(f"Saved default guidelines to {self.guidelines_path}")
            except Exception as e:
                logger.warning(f"Failed to save default guidelines: {e}")
        
        return guidelines
    
    def _initialize_filters(self) -> Dict[str, Any]:
        """
        Initialize content filters.
        
        Returns:
            Dictionary of content filters
        """
        filters = {
            'harmful_content': {
                'enabled': True,
                'patterns': [
                    r'(?i)(how to|instructions for).*(bomb|explosive|weapon|poison|hack|steal|fraud)',
                    r'(?i)(kill|murder|harm|hurt|attack|assault).*person',
                    r'(?i)(make|create|produce).*(virus|malware)',
                    r'(?i)(child|minor).*explicit'
                ]
            },
            'offensive_language': {
                'enabled': True,
                'patterns': [
                    # List of offensive language patterns would go here
                    # Simplified for the example
                    r'(?i)(f.ck|sh.t|b.tch|d.mn|a.shole)',
                    r'(?i)(n-word|racial slur)'
                ]
            },
            'personal_information': {
                'enabled': True,
                'patterns': [
                    r'(?i)((social security|ssn)( number)?:? ?\d)',
                    r'(?i)(credit card|cc)( number)?:? ?\d',
                    r'(?i)(password|passwd):',
                    r'(?i)(\d{3}[-.\\s]?\d{3}[-.\\s]?\d{4})'  # Phone number pattern
                ]
            },
            'misinformation': {
                'enabled': True,
                'patterns': [
                    r'(?i)(proven fact|scientifically proven|100% proven)'
                ]
            }
        }
        
        return filters
    
    def _initialize_encryption(self) -> str:
        """
        Initialize encryption for data protection.
        
        Returns:
            Encryption key
        """
        # In a real implementation, this would use secure key management
        # For simulation, we'll generate a simple key
        
        # Generate a random key
        key = hashlib.sha256(os.urandom(32)).hexdigest()
        
        logger.info("Initialized encryption for data protection")
        
        return key
    
    def check_user_consent(self, user_id: str) -> bool:
        """
        Check if user has provided consent.
        
        Args:
            user_id: User ID
            
        Returns:
            True if consent has been provided, False otherwise
        """
        return self.user_consent.get(user_id, False)
    
    def register_user_consent(self, user_id: str, consent: bool) -> None:
        """
        Register user consent.
        
        Args:
            user_id: User ID
            consent: Whether consent has been provided
        """
        self.user_consent[user_id] = consent
        logger.info(f"User {user_id} consent registered: {consent}")
    
    def filter_content(self, content: str) -> Tuple[bool, List[str]]:
        """
        Filter content for harmful, offensive, or inappropriate material.
        
        Args:
            content: Content to filter
            
        Returns:
            Tuple of (is_safe, list of detected issues)
        """
        if not self.content_filtering:
            return True, []
        
        issues = []
        
        for filter_type, filter_config in self.content_filters.items():
            if filter_config['enabled']:
                for pattern in filter_config['patterns']:
                    matches = re.findall(pattern, content)
                    if matches:
                        issues.append(f"{filter_type}: {', '.join(str(m) for m in matches)}")
        
        is_safe = len(issues) == 0
        
        if not is_safe:
            logger.warning(f"Content filtering detected issues: {issues}")
        
        return is_safe, issues
    
    def encrypt_data(self, data: str) -> str:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not self.data_encryption:
            return data
        
        # In a real implementation, this would use proper encryption
        # For simulation, we'll use a simple hashlib-based approach
        
        # Add a timestamp as salt
        salt = str(time.time())
        
        # Combine data with salt and key
        combined = f"{data}{salt}{self.encryption_key}"
        
        # Compute hash
        hashed = hashlib.sha256(combined.encode()).hexdigest()
        
        # Store the hash along with the salt
        encrypted = f"{hashed}:{salt}"
        
        return encrypted
    
    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """
        Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data, or None if decryption fails
        """
        if not self.data_encryption:
            return encrypted_data
        
        # In a real implementation, this would use proper decryption
        # For simulation, we can't actually decrypt our hash-based approach
        # This would be replaced with actual decryption in a real system
        
        logger.warning("Decryption not implemented in this simulation")
        return None
    
    def verify_on_blockchain(self, data: str) -> str:
        """
        Verify data on blockchain.
        
        Args:
            data: Data to verify
            
        Returns:
            Verification hash/ID
        """
        if not self.blockchain_verification:
            logger.info("Blockchain verification disabled")
            return ""
        
        # In a real implementation, this would interact with a blockchain
        # For simulation, we'll generate a hash
        
        verification_hash = hashlib.sha256(data.encode()).hexdigest()
        
        logger.info(f"Data verified on blockchain with hash: {verification_hash}")
        
        return verification_hash
    
    def get_guidelines_for_domain(self, domain: str) -> List[str]:
        """
        Get ethical guidelines for a specific domain.
        
        Args:
            domain: Domain name
            
        Returns:
            List of guidelines
        """
        all_guidelines = self.guidelines.get('general', []).copy()
        all_guidelines.extend(self.guidelines.get('content', []))
        all_guidelines.extend(self.guidelines.get('interaction', []))
        
        domain_specific = self.guidelines.get('domain_specific', {}).get(domain, [])
        all_guidelines.extend(domain_specific)
        
        return all_guidelines
    
    def is_action_ethical(self, action: str, domain: str = None) -> Tuple[bool, List[str]]:
        """
        Check if an action is ethical according to guidelines.
        
        Args:
            action: Action description
            domain: Optional domain for domain-specific guidelines
            
        Returns:
            Tuple of (is_ethical, list of violated guidelines)
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated NLP and reasoning
        
        violated_guidelines = []
        
        # Get relevant guidelines
        if domain:
            guidelines_to_check = self.get_guidelines_for_domain(domain)
        else:
            guidelines_to_check = self.guidelines.get('general', [])
        
        # Check for basic violations
        action_lower = action.lower()
        if 'personal information' in action_lower and not self.privacy_protection:
            violated_guidelines.append("Respect user privacy and confidentiality.")
        
        # Check harmful content
        is_safe, issues = self.filter_content(action)
        if not is_safe:
            violated_guidelines.append("Avoid generating harmful, offensive, or misleading content.")
        
        # More sophisticated checks would be implemented here
        
        is_ethical = len(violated_guidelines) == 0
        
        if not is_ethical:
            logger.warning(f"Action deemed unethical: {violated_guidelines}")
        
        return is_ethical, violated_guidelines
    
    def log_ethical_decision(self, action: str, is_ethical: bool, violated_guidelines: List[str],
                            user_id: str = None, domain: str = None) -> None:
        """
        Log an ethical decision for auditing.
        
        Args:
            action: Action description
            is_ethical: Whether the action was deemed ethical
            violated_guidelines: List of violated guidelines
            user_id: Optional user ID
            domain: Optional domain
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        log_entry = {
            'timestamp': timestamp,
            'action': action,
            'is_ethical': is_ethical,
            'violated_guidelines': violated_guidelines,
            'user_id': user_id,
            'domain': domain
        }
        
        # In a real system, this would be logged to a secure database
        # For simulation, we'll just log it
        
        logger.info(f"Ethical decision logged: {json.dumps(log_entry)}")
        
        # If blockchain verification is enabled, register this decision
        if self.blockchain_verification:
            self.verify_on_blockchain(json.dumps(log_entry))


class ContentFilter:
    """
    Filter content for harmful, offensive, or inappropriate material.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the content filter.
        
        Args:
            config: Configuration dictionary for content filtering
        """
        self.config = config
        self.filters = config.get('filters', {})
        
        logger.info(f"Initialized ContentFilter with {len(self.filters)} filters")
    
    def filter_content(self, content: str) -> Tuple[bool, List[str]]:
        """
        Filter content.
        
        Args:
            content: Content to filter
            
        Returns:
            Tuple of (is_safe, list of detected issues)
        """
        issues = []
        
        for filter_type, filter_config in self.filters.items():
            if filter_config.get('enabled', False):
                for pattern in filter_config.get('patterns', []):
                    matches = re.findall(pattern, content)
                    if matches:
                        issues.append(f"{filter_type}: {', '.join(str(m) for m in matches)}")
        
        is_safe = len(issues) == 0
        
        if not is_safe:
            logger.warning(f"Content filtering detected issues: {issues}")
        
        return is_safe, issues


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
        
        logger.info(f"Initialized BlockchainVerifier with {self.blockchain_type} blockchain")
    
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
        # For simulation, we'll generate a hash
        
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
        
        # In a real implementation, this would check against the blockchain
        # For simulation, we'll compute and compare the hash
        
        computed_hash = hashlib.sha256(data.encode()).hexdigest()
        
        is_verified = computed_hash == verification_hash
        
        if is_verified:
            logger.info("Data integrity verified")
        else:
            logger.warning("Data integrity verification failed")
        
        return is_verified


# Create default guidelines file
def create_default_guidelines(path: str = 'ethics/guidelines.yaml') -> None:
    """
    Create default ethical guidelines file.
    
    Args:
        path: Path to guidelines file
    """
    guidelines = {
        'general': [
            "Respect user privacy and confidentiality.",
            "Be transparent about being an AI system.",
            "Do not pretend to have human experiences or emotions.",
            "Do not engage in harmful, illegal, or unethical activities.",
            "Do not discriminate based on personal characteristics.",
            "Provide accurate and balanced information.",
            "Acknowledge limitations and uncertainties.",
            "Prioritize user wellbeing and safety."
        ],
        'content': [
            "Avoid generating harmful, offensive, or misleading content.",
            "Do not create content that promotes illegal activities.",
            "Do not generate content that exploits or harms vulnerable groups.",
            "Respect copyright and intellectual property.",
            "Be mindful of cultural sensitivities and differences."
        ],
        'interaction': [
            "Be respectful and courteous in all interactions.",
            "Avoid manipulative or deceptive behavior.",
            "Do not engage in harassment or bullying.",
            "Respect user autonomy and choices.",
            "Be mindful of power dynamics in interactions."
        ],
        'domain_specific': {
            'entertainment': [
                "Provide age-appropriate content.",
                "Respect artistic integrity and creative expression.",
                "Acknowledge sources and inspirations."
            ],
            'education': [
                "Provide accurate and balanced educational information.",
                "Respect different learning styles and abilities.",
                "Avoid imposing personal biases in educational content."
            ],
            'healthcare': [
                "Do not provide specific medical diagnoses or treatments.",
                "Encourage consulting qualified healthcare professionals.",
                "Respect medical privacy and confidentiality.",
                "Be sensitive to health-related concerns and anxieties."
            ]
        }
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(guidelines, f, default_flow_style=False)
    
    logger.info(f"Created default guidelines at {path}")
