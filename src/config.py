"""
Configuration management for the AI-based Virtual Personality System.
"""

import os
import yaml
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'system': {
        'name': 'AI Virtual Personality System',
        'version': '1.0.0',
        'log_level': 'INFO',
    },
    'data_collection': {
        'storage_path': 'data/storage',
        'max_storage_size_gb': 10,
        'edge_processing': True,
        'real_time_analysis': True,
        'sensors': {
            'audio': True,
            'video': True,
            'text': True,
            'biometric': False,
        }
    },
    'personality_generation': {
        'model_path': 'models/personality_base',
        'device': 'cuda',  # 'cpu' or 'cuda'
        'model_type': 'transformer',
        'use_2nm_ai_chips': True,
        'generation_parameters': {
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'max_length': 2048
        }
    },
    'interaction': {
        'response_time_ms': 100,
        'multimodal_fusion': True,
        'emotion_detection': True,
        'contextual_memory_turns': 10,
        'modalities': {
            'text': True,
            'voice': True,
            'vision': True,
            'gesture': False
        }
    },
    'ethics_security': {
        'content_filtering': True,
        'privacy_protection': True,
        'blockchain_verification': False,
        'ethical_guidelines_path': 'ethics/guidelines.yaml',
        'data_encryption': True,
        'user_consent_required': True,
    },
    'platforms': {
        'web': True,
        'mobile': True,
        'vr': False,
        'ar': False,
        'hologram': False,
        'smart_speaker': False,
        'cross_platform_sync': True
    },
    'learning': {
        'federated_learning': True,
        'continuous_improvement': True,
        'user_feedback_collection': True,
        'training_schedule': 'daily',
        'evaluation_metrics': ['accuracy', 'user_satisfaction', 'response_coherence']
    },
    'domains': {
        'entertainment': {
            'enabled': True,
            'default_traits': {
                'extroversion': 0.8,
                'creativity': 0.9,
                'humor': 0.85,
                'confidence': 0.9,
                'empathy': 0.6
            }
        },
        'education': {
            'enabled': True,
            'default_traits': {
                'knowledge': 0.95,
                'patience': 0.9,
                'clarity': 0.85,
                'empathy': 0.8,
                'adaptability': 0.7
            }
        },
        'healthcare': {
            'enabled': True,
            'default_traits': {
                'empathy': 0.95,
                'knowledge': 0.9,
                'calmness': 0.85,
                'trustworthiness': 0.9,
                'attentiveness': 0.8
            }
        },
        'customer_service': {
            'enabled': True,
            'default_traits': {
                'helpfulness': 0.9,
                'patience': 0.85,
                'efficiency': 0.8,
                'friendliness': 0.9,
                'knowledge': 0.75
            }
        },
        'financial': {
            'enabled': True,
            'default_traits': {
                'analytical': 0.95,
                'trustworthiness': 0.9,
                'knowledge': 0.85,
                'clarity': 0.8,
                'patience': 0.7
            }
        },
        'tourism': {
            'enabled': True,
            'default_traits': {
                'enthusiasm': 0.9,
                'knowledge': 0.85,
                'adaptability': 0.8,
                'cultural_awareness': 0.9,
                'friendliness': 0.85
            }
        }
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            # Update the default config with user-provided values
            _deep_update(config, user_config)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
    else:
        logger.warning(f"Configuration file not found at {config_path}")
        logger.info("Using default configuration")
        
        # Save the default configuration for reference
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved default configuration to {config_path}")
        except Exception as e:
            logger.warning(f"Failed to save default configuration: {e}")
    
    return config


def _deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a dictionary.
    
    Args:
        original: Original dictionary to update
        update: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for key, value in update.items():
        if key in original and isinstance(original[key], dict) and isinstance(value, dict):
            _deep_update(original[key], value)
        else:
            original[key] = value
    return original


def get_domain_config(config: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Get domain-specific configuration.
    
    Args:
        config: Main configuration dictionary
        domain: Domain name
        
    Returns:
        Domain-specific configuration
    """
    if domain in config.get('domains', {}):
        return config['domains'][domain]
    else:
        logger.warning(f"Configuration for domain '{domain}' not found")
        return {}
