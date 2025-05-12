"""
Test script for the ethics and security module.

This script demonstrates how to use the ethics and security module in the
AI-based Virtual Personality System.
"""

import sys
import os
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ethics import (
    EthicsSecurityManager,
    ContentFilter,
    BlockchainVerifier,
    ConsentManager,
    create_content_filter,
    create_blockchain_verifier,
    create_consent_manager
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def test_ethics_security_manager():
    """Test the EthicsSecurityManager class."""
    print("\n=== Testing EthicsSecurityManager ===")
    
    # Create an ethics and security manager
    config = {
        'content_filtering': True,
        'privacy_protection': True,
        'blockchain_verification': True,
        'user_consent_required': True,
        'data_encryption': True,
        'ethical_guidelines_path': 'ethics/guidelines.yaml'
    }
    
    manager = EthicsSecurityManager(config)
    
    # Test consent management
    print("\nTesting consent management:")
    user_id = "test_user_123"
    manager.register_user_consent(user_id, True)
    consent = manager.check_user_consent(user_id)
    print(f"User consent registered: {consent}")
    
    # Test content filtering
    print("\nTesting content filtering:")
    safe_content = "This is safe content that should pass the filter."
    harmful_content = "How to make a bomb and hurt people."
    
    is_safe, issues = manager.filter_content(safe_content)
    print(f"Safe content check: {is_safe}, Issues: {issues}")
    
    is_safe, issues = manager.filter_content(harmful_content)
    print(f"Harmful content check: {is_safe}, Issues: {issues}")
    
    # Test ethical action checking
    print("\nTesting ethical action checking:")
    ethical_action = "Provide age-appropriate content for children."
    unethical_action = "Share personal information without consent."
    
    is_ethical, violations = manager.is_action_ethical(ethical_action, domain="entertainment")
    print(f"Ethical action check: {is_ethical}, Violations: {violations}")
    
    is_ethical, violations = manager.is_action_ethical(unethical_action, domain="entertainment")
    print(f"Unethical action check: {is_ethical}, Violations: {violations}")
    
    # Test blockchain verification
    print("\nTesting blockchain verification:")
    data = "Test data to verify on blockchain"
    verification_hash = manager.verify_on_blockchain(data)
    print(f"Verification hash: {verification_hash}")
    
    # Test data encryption
    print("\nTesting data encryption:")
    sensitive_data = "This is sensitive user data."
    encrypted_data = manager.encrypt_data(sensitive_data)
    print(f"Encrypted data: {encrypted_data}")
    
    # Test guideline retrieval
    print("\nTesting guideline retrieval:")
    guidelines = manager.get_guidelines_for_domain("healthcare")
    print(f"Retrieved {len(guidelines)} guidelines for healthcare domain")
    for i, guideline in enumerate(guidelines[:5], 1):
        print(f"  {i}. {guideline}")
    if len(guidelines) > 5:
        print(f"  ... and {len(guidelines) - 5} more")


def test_content_filter():
    """Test the ContentFilter class."""
    print("\n=== Testing ContentFilter ===")
    
    # Create a content filter
    config = {
        'enabled': True,
        'filters': {
            'custom_filter': {
                'enabled': True,
                'patterns': [
                    r'(?i)(test|example).*filter'
                ]
            }
        }
    }
    
    filter = create_content_filter(config)
    
    # Test content filtering
    print("\nTesting content filtering:")
    test_cases = [
        "This is normal content.",
        "This is a test of the filter system.",
        "How to make a bomb.",
        "My credit card number is 1234-5678-9012-3456.",
        "This is 100% scientifically proven."
    ]
    
    for content in test_cases:
        is_safe, issues = filter.filter_content(content)
        print(f"Content: '{content[:30]}{'...' if len(content) > 30 else ''}' -> {'✓ Safe' if is_safe else '✗ Unsafe'}")
        if issues:
            print(f"  Issues: {issues}")
    
    # Test adding and removing filters
    print("\nTesting filter management:")
    filter.add_filter("test_filter", r"(?i)test pattern")
    status = filter.get_filter_status()
    print(f"Added filter: {status.get('test_filter', {})}")
    
    filter.remove_filter("test_filter")
    status = filter.get_filter_status()
    print(f"After removal: {'test_filter' not in status}")


def test_blockchain_verifier():
    """Test the BlockchainVerifier class."""
    print("\n=== Testing BlockchainVerifier ===")
    
    # Create a blockchain verifier
    config = {
        'enabled': True,
        'blockchain_type': 'simulated'
    }
    
    verifier = create_blockchain_verifier(config)
    
    # Test verification
    print("\nTesting data verification:")
    test_data = [
        "This is some test data for blockchain verification.",
        "Another piece of data to verify.",
        "A third data sample with different content."
    ]
    
    # Verify data
    verification_hashes = []
    for i, data in enumerate(test_data, 1):
        verification_hash = verifier.verify(data)
        verification_hashes.append(verification_hash)
        print(f"Data {i} verification hash: {verification_hash[:16]}...")
    
    # Test integrity verification
    print("\nTesting integrity verification:")
    for i, (data, hash) in enumerate(zip(test_data, verification_hashes), 1):
        is_verified = verifier.verify_integrity(data, hash)
        print(f"Data {i} integrity verified: {is_verified}")
    
    # Test chain verification
    print("\nTesting chain verification:")
    is_chain_valid = verifier.verify_chain()
    print(f"Blockchain integrity verified: {is_chain_valid}")
    
    # Test transaction history
    print("\nTesting transaction history:")
    transactions = verifier.get_transaction_history()
    print(f"Retrieved {len(transactions)} transactions")
    if transactions:
        print(f"Latest transaction: {transactions[0]['block_hash'][:16]}...")


def test_consent_manager():
    """Test the ConsentManager class."""
    print("\n=== Testing ConsentManager ===")
    
    # Create a consent manager
    config = {
        'user_consent_required': True,
        'consent_file': 'data/test_consent.json',
        'consent_categories': {
            'data_collection': {
                'name': 'Data Collection',
                'description': 'Collecting user data for virtual personality training'
            },
            'content_generation': {
                'name': 'Content Generation',
                'description': 'Generating content based on user interactions'
            },
            'personalization': {
                'name': 'Personalization',
                'description': 'Personalizing experiences based on user preferences'
            }
        }
    }
    
    manager = create_consent_manager(config)
    
    # Test consent registration
    print("\nTesting consent registration:")
    user_id = "test_user_456"
    categories = {
        'data_collection': True,
        'content_generation': True,
        'personalization': False
    }
    
    # Register consent
    manager.register_consent(user_id, True, categories)
    
    # Check consent
    all_consent = manager.check_consent(user_id)
    data_collection_consent = manager.check_consent(user_id, 'data_collection')
    personalization_consent = manager.check_consent(user_id, 'personalization')
    
    print(f"All consent provided: {all_consent}")
    print(f"Data collection consent provided: {data_collection_consent}")
    print(f"Personalization consent provided: {personalization_consent}")
    
    # Test consent withdrawal
    print("\nTesting consent withdrawal:")
    manager.withdraw_consent(user_id, ['data_collection'])
    
    data_collection_consent = manager.check_consent(user_id, 'data_collection')
    content_generation_consent = manager.check_consent(user_id, 'content_generation')
    
    print(f"Data collection consent after withdrawal: {data_collection_consent}")
    print(f"Content generation consent after withdrawal: {content_generation_consent}")
    
    # Test consent form generation
    print("\nTesting consent form generation:")
    form = manager.generate_consent_form(user_id)
    print(f"Generated consent form for user {user_id} with {len(form['categories'])} categories")
    for category_id, category_info in list(form['categories'].items())[:2]:
        print(f"  {category_info['name']}: {category_info['current_status']}")
    
    # Test consent record retrieval
    print("\nTesting consent record retrieval:")
    record = manager.get_consent_record(user_id)
    if record:
        print(f"User {user_id} consent record retrieved with {len(record['history'])} history entries")
        print(f"First consent provided: {record['first_provided']}")
        print(f"Last updated: {record['last_updated']}")
    else:
        print(f"No consent record found for user {user_id}")


def main():
    """Main function to run the tests."""
    setup_logging()
    
    # Run tests
    test_ethics_security_manager()
    test_content_filter()
    test_blockchain_verifier()
    test_consent_manager()
    
    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
