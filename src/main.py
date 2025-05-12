"""
Main entry point for the AI-based Virtual Personality System.
This module initializes and runs the complete system.
"""

import os
import logging
import argparse
from typing import Dict, Any

from data.collector import DataCollector
from models.personality_generator import PersonalityGenerator
from interaction.manager import InteractionManager
from ethics.security_manager import EthicsSecurityManager
from platforms.platform_manager import PlatformManager
from learning.continuous_learning import ContinuousLearningManager
from config import load_config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize all system components.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of system components
    """
    logger.info("Initializing Virtual Personality System...")
    
    # Initialize data collection and analysis module
    data_collector = DataCollector(config.get('data_collection', {}))
    logger.info("Data collection module initialized")
    
    # Initialize personality generation module
    personality_generator = PersonalityGenerator(config.get('personality_generation', {}))
    logger.info("Personality generation module initialized")
    
    # Initialize interaction management module
    interaction_manager = InteractionManager(config.get('interaction', {}))
    logger.info("Interaction management module initialized")
    
    # Initialize ethics and security module
    ethics_manager = EthicsSecurityManager(config.get('ethics_security', {}))
    logger.info("Ethics and security module initialized")
    
    # Initialize multi-platform support module
    platform_manager = PlatformManager(config.get('platforms', {}))
    logger.info("Platform support module initialized")
    
    # Initialize continuous learning module
    learning_manager = ContinuousLearningManager(config.get('learning', {}))
    logger.info("Continuous learning module initialized")
    
    return {
        'data_collector': data_collector,
        'personality_generator': personality_generator,
        'interaction_manager': interaction_manager,
        'ethics_manager': ethics_manager,
        'platform_manager': platform_manager,
        'learning_manager': learning_manager
    }


def run_system(components: Dict[str, Any]) -> None:
    """
    Run the Virtual Personality System.
    
    Args:
        components: Dictionary of system components
    """
    logger.info("Starting Virtual Personality System...")
    
    # Connect components
    components['interaction_manager'].set_personality_generator(components['personality_generator'])
    components['interaction_manager'].set_ethics_manager(components['ethics_manager'])
    components['personality_generator'].set_data_collector(components['data_collector'])
    components['personality_generator'].set_learning_manager(components['learning_manager'])
    
    # Start the platform manager, which will handle user interactions
    components['platform_manager'].start(
        interaction_manager=components['interaction_manager'],
        ethics_manager=components['ethics_manager']
    )
    
    logger.info("Virtual Personality System is running")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI-based Virtual Personality System')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize system components
    components = initialize_system(config)
    
    # Run the system
    run_system(components)


if __name__ == '__main__':
    main()
