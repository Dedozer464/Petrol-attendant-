"""
Configuration loader for Petrol Attendant Analysis
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file. Defaults to 'config.yaml' in current directory.
    
    Returns:
        Dictionary containing configuration.
    
    Raises:
        FileNotFoundError: If config file not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_output_path(config: Dict[str, Any]) -> str:
    """
    Determine output file path based on configuration.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        Full path for output file.
    """
    output_config = config.get('output', {})
    filename = output_config.get('plot_filename', 'attendants_analysis.png')
    directory = output_config.get('plot_directory', '')
    
    if directory:
        # User-specified directory
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)
    else:
        # Use current working directory
        return filename


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that configuration has required fields.
    
    Args:
        config: Configuration dictionary.
    
    Returns:
        True if valid, False otherwise.
    
    Raises:
        ValueError: If critical configuration is missing.
    """
    required_sections = ['data', 'thresholds', 'statistics', 'efficiency', 'visualization', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate efficiency weights sum to 1.0
    weights = config['efficiency']['weights']
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.001:  # Allow small float errors
        raise ValueError(f"Efficiency weights must sum to 1.0, got {total_weight}")
    
    # Validate data has attendants
    if not config['data'].get('attendants'):
        raise ValueError("No attendants in data configuration")
    
    return True
