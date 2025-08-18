#!/usr/bin/env python3
"""
Download required models for voice cloning system
Based on the guide's recommended models
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import hashlib
import logging
from typing import Optional, Dict
import torch
from TTS.api import TTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "xtts_v2": {
        "name": "XTTS-v2 Multilingual",
        "description": "Main TTS model - 16 languages, 6-second voice cloning",
        "size": "1.8GB",
        "auto_download": True,
        "tts_model_name": "tts_models/multilingual/multi-dataset/xtts_v2"
    },
    "bigvgan": {
        "name": "BigVGAN v2",
        "description": "High-quality neural vocoder from NVIDIA",
        "size": "400MB",
        "url": "https://github.com/NVIDIA/BigVGAN/releases/download/v2/bigvgan_v2_22khz_80band_256x.pt",
        "checksum": None  # Add actual checksum in production
    },
    "vocos": {
        "name": "Vocos",
        "description": "Efficient vocoder for EnCodec tokens",
        "size": "55MB",
        "auto_download": True,
        "package": "vocos"
    }
}


class ModelDownloader:
    """Handles model downloading and setup."""
    
    def __init__(self, models_dir: Path = Path("./models")):
        """Initialize model downloader."""
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No GPU detected. Models will run on CPU (slower)")
    
    def download_all(self):
        """Download all required models."""
        logger.info("Starting model downloads...")
        
        # Download XTTS-v2
        self._download_xtts_v2()
        
        # Download BigVGAN
        self._download_bigvgan()
        
        # Setup additional models
        self._setup_vocos()
        
        # Verify installations
        self._verify_models()
        
        logger.info("✅ All models downloaded successfully!")
    
    def _download_xtts_v2(self):
        """Download XTTS-v2 model using TTS library."""
        logger.info("Downloading XTTS-v2...")
        
        try:
            # TTS library handles downloading automatically
            tts = TTS(MODELS["xtts_v2"]["tts_model_name"], progress_bar=True)
            
            # Test the model
            logger.info("Testing XTTS-v2...")
            test_text = "This is a test of the voice cloning system."
            test_output = self.models_dir / "test_xtts.wav"
            
            # Use built-in speaker for testing
            tts.tts_to_file(
                text=test_text,
                language="en",
                file_path=str(test_output)
            )
            
            if test_output.exists():
                logger.info("✅ XTTS-v2 installed and working!")
                test_output.unlink()  # Clean up test file
            else:
                raise Exception("Test synthesis failed")
                
        except Exception as e:
            logger.error(f"Failed to download XTTS-v2: {e}")
            sys.exit(1)
    
    def _download_bigvgan(self):
        """Download BigVGAN vocoder."""
        logger.info("Downloading BigVGAN v2...")
        
        model_info = MODELS["bigvgan"]
        output_path = self.models_dir / "bigvgan_v2" / "checkpoint.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            logger.info("BigVGAN already downloaded")
            return
        
        try:
            # Download with progress bar
            response = requests.get(model_info["url"], stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(
                    desc="BigVGAN",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            logger.info("✅ BigVGAN downloaded successfully!")
            
            # Download config file
            config_url = "https://raw.githubusercontent.com/NVIDIA/BigVGAN/main/configs/bigvgan_v2_22khz_80band_256x.json"
            config_path = output_path.parent / "config.json"
            
            logger.info("Downloading BigVGAN config...")
            config_response = requests.get(config_url)
            config_response.raise_for_status()
            
            with open(config_path, 'w') as f:
                f.write(config_response.text)
            
        except Exception as e:
            logger.error(f"Failed to download BigVGAN: {e}")
            if output_path.exists():
                output_path.unlink()
            sys.exit(1)
    
    def _setup_vocos(self):
        """Setup Vocos vocoder."""
        logger.info("Setting up Vocos...")
        
        try:
            import vocos
            logger.info("✅ Vocos already installed")
        except ImportError:
            logger.info("Installing Vocos...")
            os.system(f"{sys.executable} -m pip install vocos")
            
            # Verify installation
            try:
                import vocos
                logger.info("✅ Vocos installed successfully!")
            except ImportError:
                logger.error("Failed to install Vocos")
                sys.exit(1)
    
    def _download_file(
        self,
        url: str,
        output_path: Path,
        checksum: Optional[str] = None
    ) -> bool:
        """
        Download file with progress bar and checksum verification.
        
        Args:
            url: Download URL
            output_path: Where to save the file
            checksum: Optional SHA256 checksum
            
        Returns:
            True if successful
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f:
                with tqdm(
                    desc=output_path.name,
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            # Verify checksum if provided
            if checksum:
                file_hash = self._calculate_checksum(output_path)
                if file_hash != checksum:
                    logger.error(f"Checksum mismatch for {output_path.name}")
                    output_path.unlink()
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _verify_models(self):
        """Verify all models are properly installed."""
        logger.info("\nVerifying model installations...")
        
        issues = []
        
        # Check XTTS-v2
        try:
            from TTS.api import TTS
            tts = TTS(MODELS["xtts_v2"]["tts_model_name"])
            logger.info("✅ XTTS-v2: OK")
        except Exception as e:
            issues.append(f"XTTS-v2: {e}")
        
        # Check BigVGAN
        bigvgan_path = self.models_dir / "bigvgan_v2" / "checkpoint.pt"
        if bigvgan_path.exists():
            logger.info("✅ BigVGAN: OK")
        else:
            issues.append("BigVGAN: Not found")
        
        # Check Vocos
        try:
            import vocos
            logger.info("✅ Vocos: OK")
        except ImportError:
            issues.append("Vocos: Not installed")
        
        if issues:
            logger.error("\n❌ Issues found:")
            for issue in issues:
                logger.error(f"  - {issue}")
            sys.exit(1)
        else:
            logger.info("\n✅ All models verified successfully!")
    
    def download_optional_models(self):
        """Download optional models for advanced features."""
        logger.info("\nDownloading optional models...")
        
        optional_models = {
            "emotion_model": {
                "name": "Emotion Control Model",
                "description": "For emotion control in synthesis",
                "url": None,  # Add URL when available
                "size": "200MB"
            },
            "denoiser": {
                "name": "Speech Enhancement Model",
                "description": "For audio quality enhancement",
                "url": None,  # Add URL when available
                "size": "150MB"
            }
        }
        
        logger.info("Optional models not yet implemented")
        # Implement when models become available


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download models for voice cloning system"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("./models"),
        help="Directory to store models"
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Also download optional models"
    )
    
    args = parser.parse_args()
    
    # Print system info
    logger.info("System Information:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")
    
    # Download models
    downloader = ModelDownloader(args.models_dir)
    downloader.download_all()
    
    if args.optional:
        downloader.download_optional_models()
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Model setup complete!")
    logger.info("You can now run the voice cloning system.")
    logger.info("="*50)


if __name__ == "__main__":
    main()
