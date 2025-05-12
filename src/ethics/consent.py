"""
User consent management module for the AI-based Virtual Personality System.

This module provides functionality for managing user consent in the virtual personality
system, ensuring that users have provided explicit consent for data collection, processing,
and other aspects of the system.
"""

import time
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple
import os

logger = logging.getLogger(__name__)


class ConsentManager:
    """
    Manages user consent for the virtual personality system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the consent manager.
        
        Args:
            config: Configuration dictionary for consent management
        """
        self.config = config
        self.required = config.get('user_consent_required', True)
        
        # Dictionary to store user consent records
        self.user_consent = {}
        
        # Consent file path
        self.consent_file = config.get('consent_file', 'data/consent.json')
        
        # Define consent categories
        self.consent_categories = config.get('consent_categories', {
            'data_collection': {
                'name': 'Data Collection',
                'description': 'Collecting user data for virtual personality training and operation'
            },
            'content_generation': {
                'name': 'Content Generation',
                'description': 'Generating content and responses based on user interactions'
            },
            'data_sharing': {
                'name': 'Data Sharing',
                'description': 'Sharing anonymized data for system improvement'
            },
            'personalization': {
                'name': 'Personalization',
                'description': 'Personalizing virtual personality behavior based on user preferences'
            },
            'third_party': {
                'name': 'Third Party Services',
                'description': 'Using third party services for enhanced functionality'
            }
        })
        
        # Load existing consent records if available
        self._load_consent_records()
        
        logger.info(f"Initialized ConsentManager with consent required: {self.required}")
        logger.info(f"Consent categories: {list(self.consent_categories.keys())}")
    
    def _load_consent_records(self) -> None:
        """
        Load existing consent records from file.
        """
        if os.path.exists(self.consent_file):
            try:
                with open(self.consent_file, 'r') as f:
                    self.user_consent = json.load(f)
                logger.info(f"Loaded {len(self.user_consent)} consent records")
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")
        else:
            logger.info("No existing consent records found")
    
    def _save_consent_records(self) -> None:
        """
        Save consent records to file.
        """
        try:
            os.makedirs(os.path.dirname(self.consent_file), exist_ok=True)
            with open(self.consent_file, 'w') as f:
                json.dump(self.user_consent, f, indent=2)
            logger.info(f"Saved {len(self.user_consent)} consent records")
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")
    
    def check_consent(self, user_id: str, category: str = None) -> bool:
        """
        Check if user has provided consent.
        
        Args:
            user_id: User ID
            category: Optional consent category
            
        Returns:
            True if consent has been provided, False otherwise
        """
        if not self.required:
            return True
        
        if user_id not in self.user_consent:
            return False
        
        if category:
            return self.user_consent[user_id].get('categories', {}).get(category, False)
        else:
            return self.user_consent[user_id].get('all_provided', False)
    
    def register_consent(self, user_id: str, consent: bool, categories: Dict[str, bool] = None,
                         metadata: Dict[str, Any] = None) -> None:
        """
        Register user consent.
        
        Args:
            user_id: User ID
            consent: Whether consent has been provided for all categories
            categories: Optional dictionary of category-specific consent
            metadata: Optional additional metadata about the consent
        """
        timestamp = time.time()
        
        if categories is None:
            categories = {category: consent for category in self.consent_categories}
        
        if metadata is None:
            metadata = {}
        
        # Create or update consent record
        if user_id not in self.user_consent:
            self.user_consent[user_id] = {
                'first_provided': timestamp,
                'last_updated': timestamp,
                'all_provided': consent,
                'categories': categories,
                'metadata': metadata,
                'history': [{
                    'timestamp': timestamp,
                    'all_provided': consent,
                    'categories': categories.copy(),
                    'metadata': metadata.copy()
                }]
            }
        else:
            # Update existing record
            self.user_consent[user_id]['last_updated'] = timestamp
            self.user_consent[user_id]['all_provided'] = consent
            
            # Update categories
            for category, value in categories.items():
                self.user_consent[user_id]['categories'][category] = value
            
            # Update metadata
            for key, value in metadata.items():
                self.user_consent[user_id]['metadata'][key] = value
            
            # Add to history
            self.user_consent[user_id]['history'].append({
                'timestamp': timestamp,
                'all_provided': consent,
                'categories': categories.copy(),
                'metadata': metadata.copy()
            })
        
        # Save updated records
        self._save_consent_records()
        
        logger.info(f"User {user_id} consent registered: all={consent}, categories={categories}")
    
    def withdraw_consent(self, user_id: str, categories: List[str] = None,
                        metadata: Dict[str, Any] = None) -> None:
        """
        Withdraw user consent.
        
        Args:
            user_id: User ID
            categories: Optional list of categories to withdraw consent for
            metadata: Optional additional metadata about the consent withdrawal
        """
        if user_id not in self.user_consent:
            logger.warning(f"No consent record found for user {user_id}")
            return
        
        timestamp = time.time()
        
        if categories is None:
            # Withdraw all consent
            self.user_consent[user_id]['all_provided'] = False
            for category in self.user_consent[user_id]['categories']:
                self.user_consent[user_id]['categories'][category] = False
        else:
            # Withdraw specific categories
            for category in categories:
                if category in self.user_consent[user_id]['categories']:
                    self.user_consent[user_id]['categories'][category] = False
            
            # Check if any categories still have consent
            all_withdrawn = all(not value for value in self.user_consent[user_id]['categories'].values())
            if all_withdrawn:
                self.user_consent[user_id]['all_provided'] = False
        
        # Update last updated timestamp
        self.user_consent[user_id]['last_updated'] = timestamp
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                self.user_consent[user_id]['metadata'][key] = value
        
        # Add to history
        self.user_consent[user_id]['history'].append({
            'timestamp': timestamp,
            'action': 'withdraw',
            'categories': categories,
            'all_provided': self.user_consent[user_id]['all_provided'],
            'metadata': metadata.copy() if metadata else {}
        })
        
        # Save updated records
        self._save_consent_records()
        
        if categories:
            logger.info(f"User {user_id} withdrew consent for categories: {categories}")
        else:
            logger.info(f"User {user_id} withdrew all consent")
    
    def get_consent_record(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the consent record for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Consent record dictionary or None if not found
        """
        return self.user_consent.get(user_id)
    
    def get_consent_status(self, user_id: str) -> Dict[str, Any]:
        """
        Get the current consent status for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with consent status information
        """
        if user_id not in self.user_consent:
            return {
                'all_provided': False,
                'categories': {category: False for category in self.consent_categories}
            }
        
        record = self.user_consent[user_id]
        
        return {
            'all_provided': record['all_provided'],
            'categories': record['categories'],
            'last_updated': record['last_updated']
        }
    
    def generate_consent_form(self, user_id: str, include_categories: List[str] = None) -> Dict[str, Any]:
        """
        Generate a consent form for a user.
        
        Args:
            user_id: User ID
            include_categories: Optional list of categories to include
            
        Returns:
            Dictionary representing a consent form
        """
        if include_categories is None:
            categories = self.consent_categories
        else:
            categories = {k: v for k, v in self.consent_categories.items() if k in include_categories}
        
        # Get current consent status
        current_status = self.get_consent_status(user_id)
        
        form = {
            'user_id': user_id,
            'timestamp': time.time(),
            'form_id': hashlib.sha256(f"{user_id}:{time.time()}".encode()).hexdigest()[:16],
            'categories': {},
            'current_status': current_status
        }
        
        for category_id, category_info in categories.items():
            form['categories'][category_id] = {
                'name': category_info['name'],
                'description': category_info['description'],
                'current_status': current_status['categories'].get(category_id, False)
            }
        
        return form


# Create a factory function to make it easy to create consent managers
def create_consent_manager(config: Dict[str, Any] = None) -> ConsentManager:
    """
    Create a consent manager.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ConsentManager instance
    """
    if config is None:
        config = {'user_consent_required': True}
    
    return ConsentManager(config)
