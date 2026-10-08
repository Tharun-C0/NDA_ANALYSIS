# =============================================================================
# NDA Research Project - Configuration Loader
# =============================================================================
# Provides a single entry point for loading the YAML configuration.
# All modules import config through this utility to ensure consistency.
# =============================================================================

import os
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """
    Returns the project root directory.
    
    Determines root by walking up from this file's location until we find
    the configs/ directory. This makes the project portable across machines.
    """
    current = Path(__file__).resolve().parent
    # Walk up until we find configs/config.yaml
    for _ in range(10):  # Safety limit
        if (current / "configs" / "config.yaml").exists():
            return current
        current = current.parent
    # Fallback: assume two levels up from configs/
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> dict:
    """
    Load the YAML configuration file.
    
    Args:
        config_path: Optional explicit path to config.yaml.
                     If None, uses the default location at configs/config.yaml
                     relative to project root.
    
    Returns:
        dict: Parsed configuration dictionary.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is malformed.
    """
    if config_path is None:
        root = get_project_root()
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected location: configs/config.yaml relative to project root."
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths to absolute paths based on project root
    root = get_project_root()
    config["_project_root"] = str(root)

    # Helper to resolve a path if relative
    def _resolve(path_str):
        if not os.path.isabs(path_str):
            return str(root / path_str)
        return path_str

    # Resolve dataset paths
    for key in ["raw_dir", "processed_dir", "splits_dir", "external_dir", "documents_dir"]:
        if key in config.get("dataset", {}):
            config["dataset"][key] = _resolve(config["dataset"][key])

    # Resolve output paths
    for key in ["reports_dir", "models_dir", "experiments_dir", "figures_dir"]:
        if key in config.get("output", {}):
            config["output"][key] = _resolve(config["output"][key])

    # Resolve extraction paths
    if "output_dir" in config.get("extraction", {}):
        config["extraction"]["output_dir"] = _resolve(config["extraction"]["output_dir"])

    # Resolve segmentation paths
    if "output_dir" in config.get("segmentation", {}):
        config["segmentation"]["output_dir"] = _resolve(config["segmentation"]["output_dir"])

    # Resolve log directory
    if "log_dir" in config.get("logging", {}):
        config["logging"]["log_dir"] = _resolve(config["logging"]["log_dir"])

    return config

