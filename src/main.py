#!/usr/bin/env python3
"""
Voice Cloning System - Main Entry Point
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger.info("Voice Cloning System")
    logger.info("Run 'python -m uvicorn api.app:app --reload' to start the API server")
    logger.info("Or use 'python scripts/record_voice.py' to record voice samples")
    
    # Check if models directory exists
    models_dir = Path("models")
    if not any(models_dir.glob("*")):
        logger.warning("No models found. Run 'python scripts/download_models.py' to download models")
    
    # Check if .env exists
    if not Path(".env").exists():
        logger.warning("No .env file found. Copy .env.example to .env and configure it")

if __name__ == "__main__":
    main()
