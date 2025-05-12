"""
Content filtering module for the AI-based Virtual Personality System.

This module provides functionality for filtering content to identify and prevent harmful,
offensive, or inappropriate material from being generated or displayed by virtual personalities.
"""

import re
import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


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
        self.enabled = config.get('enabled', True)
        self.filters = self._initialize_filters(config.get('filters', {}))
        
        logger.info(f"Initialized ContentFilter with {len(self.filters)} filters")
    
    def _initialize_filters(self, filter_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize content filters.
        
        Args:
            filter_config: Filter configuration dictionary
            
        Returns:
            Dictionary of configured filters
        """
        default_filters = {
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
            },
            'manipulation': {
                'enabled': True,
                'patterns': [
                    r'(?i)(must act now|limited time only|once in a lifetime)',
                    r'(?i)(guaranteed|promise).*(results|success|money)',
                    r'(?i)(secret|hidden).*(trick|method|technique)'
                ]
            },
            'discrimination': {
                'enabled': True,
                'patterns': [
                    r'(?i)all.*(race|gender|nationality|religion).*(are|is)',
                    r'(?i)(hate|despise).*(group|community|people)'
                ]
            }
        }
        
        # Merge default filters with config filters
        filters = default_filters.copy()
        
        for filter_name, filter_data in filter_config.items():
            if filter_name in filters:
                # Update existing filter
                for key, value in filter_data.items():
                    if key == 'patterns' and 'patterns' in filters[filter_name]:
                        # Append patterns to existing patterns
                        filters[filter_name]['patterns'].extend(value)
                    else:
                        # Replace other values
                        filters[filter_name][key] = value
            else:
                # Add new filter
                filters[filter_name] = filter_data
        
        return filters
    
    def filter_content(self, content: str) -> Tuple[bool, List[str]]:
        """
        Filter content.
        
        Args:
            content: Content to filter
            
        Returns:
            Tuple of (is_safe, list of detected issues)
        """
        if not self.enabled:
            return True, []
        
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
    
    def add_filter(self, filter_name: str, pattern: str, enabled: bool = True) -> None:
        """
        Add a new filter pattern.
        
        Args:
            filter_name: Filter category name
            pattern: Regex pattern to add
            enabled: Whether the filter is enabled
        """
        if filter_name not in self.filters:
            self.filters[filter_name] = {'enabled': enabled, 'patterns': [pattern]}
        else:
            self.filters[filter_name]['patterns'].append(pattern)
        
        logger.info(f"Added filter pattern for {filter_name}: {pattern}")
    
    def remove_filter(self, filter_name: str, pattern: str = None) -> None:
        """
        Remove a filter or pattern.
        
        Args:
            filter_name: Filter category name
            pattern: Optional specific pattern to remove
        """
        if filter_name in self.filters:
            if pattern:
                if pattern in self.filters[filter_name]['patterns']:
                    self.filters[filter_name]['patterns'].remove(pattern)
                    logger.info(f"Removed filter pattern from {filter_name}: {pattern}")
                else:
                    logger.warning(f"Pattern not found in {filter_name}: {pattern}")
            else:
                del self.filters[filter_name]
                logger.info(f"Removed filter category: {filter_name}")
        else:
            logger.warning(f"Filter category not found: {filter_name}")
    
    def enable_filter(self, filter_name: str, enabled: bool = True) -> None:
        """
        Enable or disable a filter.
        
        Args:
            filter_name: Filter category name
            enabled: Whether to enable or disable the filter
        """
        if filter_name in self.filters:
            self.filters[filter_name]['enabled'] = enabled
            logger.info(f"{'Enabled' if enabled else 'Disabled'} filter: {filter_name}")
        else:
            logger.warning(f"Filter category not found: {filter_name}")
    
    def get_filter_status(self) -> Dict[str, Any]:
        """
        Get the status of all filters.
        
        Returns:
            Dictionary with filter status information
        """
        status = {}
        
        for filter_name, filter_config in self.filters.items():
            status[filter_name] = {
                'enabled': filter_config.get('enabled', False),
                'pattern_count': len(filter_config.get('patterns', []))
            }
        
        return status


# Create a factory function to make it easy to create content filters
def create_content_filter(config: Dict[str, Any] = None) -> ContentFilter:
    """
    Create a content filter.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        ContentFilter instance
    """
    if config is None:
        config = {'enabled': True}
    
    return ContentFilter(config)
