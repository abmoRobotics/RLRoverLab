#!/usr/bin/env python3
"""
Model downloader utilities for RLRoverLab.
Handles automatic downloading and caching of Cosmos tokenizer models.
"""

import os
from pathlib import Path
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Get the cache directory for RLRoverLab models."""
    cache_dir = Path.home() / ".cache" / "rlroverlab" / "models" / "pretrained_checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_cosmos_models(
    models: Optional[List[str]] = None,
    force_download: bool = False,
    token: Optional[str] = None
) -> bool:
    """Download Cosmos tokenizer models from Hugging Face.
    
    Args:
        models: List of model names to download. If None, downloads all default models.
        force_download: If True, re-download even if models exist.
        token: Hugging Face token. If None, will try to get from environment or prompt user.
    
    Returns:
        True if all downloads were successful, False otherwise.
    """
    try:
        from huggingface_hub import login, snapshot_download
    except ImportError:
        logger.error("huggingface_hub not installed. Please run: pip install huggingface_hub")
        return False
    
    # Default model names
    if models is None:
        models = [
            "Cosmos-0.1-Tokenizer-CI8x8",
            "Cosmos-0.1-Tokenizer-CI16x16", 
            "Cosmos-0.1-Tokenizer-DI8x8",
            "Cosmos-0.1-Tokenizer-DI16x16",
        ]
    
    # Handle authentication
    if token is None:
        token = os.environ.get("HUGGINGFACE_TOKEN")
    
    # Keep asking for token until authentication succeeds or user cancels
    while True:
        if token is None:
            try:
                token = input("Please enter your Hugging Face API token: ")
                if not token.strip():
                    logger.warning("Empty token provided. Please enter a valid token.")
                    token = None
                    continue
            except (KeyboardInterrupt, EOFError):
                logger.error("Authentication cancelled by user.")
                return False
        
        try:
            login(token=token, add_to_git_credential=True)
            logger.info("Successfully authenticated with Hugging Face!")
            break  # Authentication successful, exit the loop
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            logger.warning("Please check your token and try again.")
            token = None  # Reset token to ask again
            
            # Ask if user wants to retry or cancel
            try:
                retry = input("Would you like to try again? (y/n): ").lower().strip()
                if retry not in ['y', 'yes']:
                    logger.error("Authentication cancelled by user.")
                    return False
            except (KeyboardInterrupt, EOFError):
                logger.error("Authentication cancelled by user.")
                return False
    
    # Get cache directory
    cache_dir = get_cache_dir()
    
    # Download each model
    success_count = 0
    for model_name in models:
        model_dir = cache_dir / model_name
        
        # Check if model already exists
        if model_dir.exists() and not force_download:
            required_files = ["encoder.jit", "decoder.jit", "config.json"]
            if all((model_dir / file).exists() for file in required_files):
                logger.info(f"✓ {model_name} already exists in cache")
                success_count += 1
                continue
        
        # Download model
        hf_repo = f"nvidia/{model_name}"
        logger.info(f"Downloading {model_name}...")
        
        try:
            snapshot_download(
                repo_id=hf_repo,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False  # Use actual files instead of symlinks
            )
            logger.info(f"✓ Successfully downloaded {model_name}")
            success_count += 1
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")
    
    all_successful = success_count == len(models)
    if all_successful:
        logger.info(f"All {len(models)} models downloaded successfully to {cache_dir}")
    else:
        logger.warning(f"Downloaded {success_count}/{len(models)} models successfully")
    
    return all_successful


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a specific model if it exists in cache.
    
    Args:
        model_name: Name of the Cosmos tokenizer model.
    
    Returns:
        Path to the model directory if it exists, None otherwise.
    """
    cache_dir = get_cache_dir()
    model_dir = cache_dir / model_name
    
    if model_dir.exists():
        # Check if required files exist
        required_files = ["encoder.jit", "decoder.jit", "config.json"]
        if all((model_dir / file).exists() for file in required_files):
            return model_dir
    
    return None


def ensure_model_available(model_name: str, auto_download: bool = True) -> Optional[Path]:
    """Ensure a model is available, downloading if necessary.
    
    Args:
        model_name: Name of the Cosmos tokenizer model.
        auto_download: If True, automatically download the model if not found.
    
    Returns:
        Path to the model directory if available, None otherwise.
    """
    model_path = get_model_path(model_name)
    
    if model_path is not None:
        return model_path
    
    if auto_download:
        logger.info(f"Model {model_name} not found in cache. Attempting to download...")
        if download_cosmos_models(models=[model_name]):
            return get_model_path(model_name)
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Cosmos tokenizer models")
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific models to download"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-download even if models exist"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face API token"
    )
    
    args = parser.parse_args()
    
    success = download_cosmos_models(
        models=args.models,
        force_download=args.force,
        token=args.token
    )
    
    exit(0 if success else 1)
