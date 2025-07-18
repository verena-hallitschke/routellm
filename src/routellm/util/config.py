"""Module to manage configuration settings for the RouteLLM project."""

import json
from typing import Any


# Designed as singleton
class Config:
    """Configuration manager for the RouteLLM project."""

    REQUIRED_CONFIG_KEYS = [
        "azure-subscription-key",
    ]

    def __new__(cls):
        """Create a singleton instance of Config."""
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
        return cls.instance

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        self.asset_path = "assets"
        self.config_file_path = "config.json"
        with open(self.config_file_path, "rt") as config_file:
            config = json.load(config_file)

        for key in Config.REQUIRED_CONFIG_KEYS:
            if config.get(key) is None:
                raise ValueError(f"Missing value in '{self.config_file_path}': '{key}'")

        self._values = config

    def get(self, key: str) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): Key of the configuration value.

        Returns:
            Any: Associated value for the key, or None if the key does not exist.

        """
        return self._values.get(key)
