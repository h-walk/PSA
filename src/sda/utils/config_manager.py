"""
Configuration management module for SDA.

This module provides functionality for loading, validating, and managing
configuration settings for SDA analysis.
"""
import yaml
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union
import json

logger = logging.getLogger(__name__)

class ConfigManager:
    """Class for managing SDA configuration settings."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file (optional)
        """
        self.config: Dict[str, Any] = {}
        if config_file is not None:
            self.load_config(config_file)
            
    def load_config(self, config_file: Union[str, Path]) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the configuration file
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_keys = [
            'trajectory',
            'analysis',
            'output'
        ]
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        # Validate trajectory settings
        traj_keys = ['file', 'format', 'dt']
        for key in traj_keys:
            if key not in self.config['trajectory']:
                raise ValueError(f"Missing required trajectory setting: {key}")
                
        # Validate analysis settings
        analysis_keys = ['k_points', 'frequencies', 'window_size']
        for key in analysis_keys:
            if key not in self.config['analysis']:
                raise ValueError(f"Missing required analysis setting: {key}")
                
        # Validate output settings
        output_keys = ['directory', 'format']
        for key in output_keys:
            if key not in self.config['output']:
                raise ValueError(f"Missing required output setting: {key}")
                
    def get_trajectory_config(self) -> Dict[str, Any]:
        """
        Get trajectory configuration settings.
        
        Returns:
            Dictionary of trajectory settings
        """
        return self.config.get('trajectory', {})
        
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get analysis configuration settings.
        
        Returns:
            Dictionary of analysis settings
        """
        return self.config.get('analysis', {})
        
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get output configuration settings.
        
        Returns:
            Dictionary of output settings
        """
        return self.config.get('output', {})
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration settings.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def _update_dict(d: Dict[str, Any], u: Dict[str, Any]) -> None:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    _update_dict(d[k], v)
                else:
                    d[k] = v
                    
        _update_dict(self.config, updates)
        self._validate_config()
        
    def save_config(self, output_file: Union[str, Path]) -> None:
        """
        Save current configuration to a file.
        
        Args:
            output_file: Path to save the configuration to
        """
        output_path = Path(output_file)
        logger.info(f"Saving configuration to {output_path}")
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the current configuration as a dictionary.
        
        Returns:
            Dictionary of current configuration
        """
        return self.config.copy()
        
    def to_json(self) -> str:
        """
        Get the current configuration as a JSON string.
        
        Returns:
            JSON string representation of configuration
        """
        return json.dumps(self.config, indent=4)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """
        Create a ConfigManager instance from a dictionary.
        
        Args:
            config_dict: Dictionary of configuration settings
            
        Returns:
            ConfigManager instance
        """
        instance = cls()
        instance.config = config_dict
        instance._validate_config()
        return instance 