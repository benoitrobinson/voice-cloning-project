#!/usr/bin/env python3
"""
Automated Voice Cloning Project Setup Script
This script creates the complete project structure and all necessary files
"""

import os
import sys
from pathlib import Path
import base64
import subprocess

# Define the project structure
PROJECT_NAME = "voice-cloning-project"

def create_directory_structure():
    """Create the complete directory structure."""
    directories = [
        "config",
        "src", 
        "models",
        "data/recordings/raw",
        "data/recordings/processed",
        "data/scripts",
        "data/voices",
        "data/consent_forms",
        "utils",
        "api/routes",
        "api/middleware",
        "tests/fixtures",
        "scripts",
        "notebooks",
        "docs",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Add __init__.py to Python packages
        if not directory.startswith(("data", "logs", "docs", "notebooks")):
            init_file = Path(directory) / "__init__.py"
            init_file.touch()
    
    print("✓ Directory structure created")

def create_requirements_txt():
    """Create requirements.txt file."""
    content = """# Core TTS Libraries
TTS==0.22.0  # Coqui TTS with XTTS-v2
torch>=2.0.0
torchaudio>=2.0.0
transformers>=4.35.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
webrtcvad>=2.0.10
noisereduce>=3.0.0
pedalboard>=0.8.0

# Vocoders
vocos>=0.1.0

# Quality Enhancement
speechbrain>=1.0.0
asteroid>=0.6.0
demucs>=4.0.0

# API & Web
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
python-multipart>=0.0.6

# Utilities
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.65.0

# Security & Auth
cryptography>=41.0.0
PyJWT>=2.8.0
passlib>=1.7.4

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0

# Monitoring
prometheus-client>=0.18.0

# Audio recording
sounddevice>=0.4.6

# Optional but recommended
pesq>=0.0.4
pyloudnorm>=0.1.0
"""
    with open("requirements.txt", "w") as f:
        f.write(content)
    print("✓ requirements.txt created")

def create_env_example():
    """Create .env.example file."""
    content = """# Environment Configuration
ENVIRONMENT=development
DEBUG=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secure-api-key-here

# Model Paths
MODEL_PATH=./models/xtts_v2
VOCODER_PATH=./models/bigvgan_v2
CHECKPOINT_DIR=./models/checkpoints

# Audio Settings
SAMPLE_RATE=22050
MAX_AUDIO_LENGTH=600  # seconds
MIN_AUDIO_LENGTH=3    # seconds

# GPU Settings
CUDA_VISIBLE_DEVICES=0
USE_GPU=true
BATCH_SIZE=8

# Security
JWT_SECRET=your-jwt-secret-here
CONSENT_ENCRYPTION_KEY=your-encryption-key-here

# Storage
VOICE_STORAGE_PATH=./data/voices
RECORDING_PATH=./data/recordings
MAX_STORAGE_GB=100

# Legal Compliance
REQUIRE_CONSENT=true
CONSENT_EXPIRY_DAYS=365
WATERMARK_ENABLED=true
"""
    with open(".env.example", "w") as f:
        f.write(content)
    print("✓ .env.example created")

def create_docker_compose():
    """Create docker-compose.yml file."""
    content = """version: '3.8'

services:
  voice-cloning-api:
    build: .
    container_name: voice-cloning-system
    ports:
      - "${API_PORT:-8000}:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-production}
      - API_KEY=${API_KEY}
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: voice-cloning-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    container_name: voice-cloning-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

volumes:
  redis-data:
  prometheus-data:
"""
    with open("docker-compose.yml", "w") as f:
        f.write(content)
    print("✓ docker-compose.yml created")

def create_gitignore():
    """Create .gitignore file."""
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv/

# IDE
.idea/
.vscode/
*.code-workspace

# Environment variables
.env
.env.local

# Models (too large for git)
models/
*.pt
*.pth
*.ckpt
*.bin

# Audio files
*.wav
*.mp3
*.ogg
*.flac

# Data
data/recordings/
data/voices/
data/consent_forms/*.pdf

# Logs
logs/
*.log

# Database
*.db
*.sqlite

# Cache
.cache/
.pytest_cache/
.coverage

# OS
.DS_Store
Thumbs.db

# Temporary
*.tmp
*.swp
*~

# Secrets
*.pem
*.key
.encryption_key
"""
    with open(".gitignore", "w") as f:
        f.write(content)
    print("✓ .gitignore created")

def create_readme():
    """Create basic README.md file."""
    content = """# Voice Cloning System 🎤

A state-of-the-art voice cloning system implementing the latest techniques from 2024-2025.

## Quick Start

1. **Setup Environment**
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Configure Settings**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Download Models**
   ```bash
   python scripts/download_models.py
   ```

4. **Start API Server**
   ```bash
   python -m uvicorn api.app:app --reload
   ```

## Documentation

- API Docs: http://localhost:8000/docs
- Full documentation in `docs/` directory

## Features

- 🌍 Multilingual support (16+ languages)
- ⚡ 3-second voice cloning
- 🎭 Emotion control
- 🔒 Ethical safeguards
- 📊 Quality enhancement

## License

MIT License - see LICENSE file
"""
    with open("README.md", "w") as f:
        f.write(content)
    print("✓ README.md created")

def create_setup_environment_script():
    """Create the setup_environment.sh script."""
    content = '''#!/bin/bash
# Setup script for Voice Cloning Project

echo "🎤 Setting up Voice Cloning Environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.8+ is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{recordings/{raw,processed},voices,scripts,consent_forms}
mkdir -p models/{checkpoints}
mkdir -p logs
mkdir -p notebooks

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
fi

echo "✅ Setup complete! Activate the virtual environment with:"
echo "   source venv/bin/activate"
'''
    
    script_path = Path("scripts/setup_environment.sh")
    with open(script_path, "w") as f:
        f.write(content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print("✓ setup_environment.sh created")

def create_main_py():
    """Create a minimal main.py to get started."""
    content = '''#!/usr/bin/env python3
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
'''
    
    with open("src/main.py", "w") as f:
        f.write(content)
    print("✓ src/main.py created")

def create_minimal_stubs():
    """Create minimal stub files to prevent import errors."""
    
    # Create minimal settings.py
    settings_content = '''"""Configuration settings"""
from pathlib import Path

class Settings:
    def __init__(self):
        self.sample_rate = 22050
        self.min_audio_length = 3
        self.max_audio_length = 600
        self.voice_storage_path = Path("./data/voices")
        self.use_gpu = True
        self.api_key = "your-api-key"
        self.api_host = "0.0.0.0"
        self.api_port = 8000
'''
    with open("config/settings.py", "w") as f:
        f.write(settings_content)
    
    # Create minimal voice_cloner.py stub
    cloner_content = '''"""Voice Cloner Module - Install TTS library to use"""
import logging
logger = logging.getLogger(__name__)

class VoiceCloner:
    def __init__(self, settings):
        logger.warning("VoiceCloner: TTS library not installed. Run pip install -r requirements.txt")
        self.settings = settings
'''
    with open("src/voice_cloner.py", "w") as f:
        f.write(cloner_content)
    
    print("✓ Minimal stub files created")

def create_recording_script():
    """Create the phonetic recording script."""
    content = """# Voice Recording Script for Optimal Cloning

## Instructions
- Read each sentence naturally at your normal speaking pace
- Pause for 1-2 seconds between sentences
- Maintain consistent distance from microphone (6-8 inches)

## Sentences

1. The quick brown fox jumps over the lazy dog near the riverbank.
2. She sells seashells by the seashore on sunny summer days.
3. Peter Piper picked a peck of pickled peppers perfectly.
4. Technology advances rapidly in the modern digital age.
5. Please record your voice clearly and naturally.
"""
    with open("data/scripts/recording_script.md", "w") as f:
        f.write(content)
    print("✓ Recording script created")

def main():
    """Main setup function."""
    print("🚀 Voice Cloning Project Setup")
    print("="*40)
    
    # Check if we're in the right directory
    if Path("voice-cloning-project").exists():
        response = input("Directory 'voice-cloning-project' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return
    
    # Create project directory
    if not Path.cwd().name == "voice-cloning-project":
        Path("voice-cloning-project").mkdir(exist_ok=True)
        os.chdir("voice-cloning-project")
    
    # Create all components
    print("\nCreating project structure...")
    create_directory_structure()
    create_requirements_txt()
    create_env_example()
    create_docker_compose()
    create_gitignore()
    create_readme()
    create_setup_environment_script()
    create_main_py()
    create_minimal_stubs()
    create_recording_script()
    
    print("\n✅ Project setup complete!")
    print("\n📋 Next steps:")
    print("1. cd voice-cloning-project")
    print("2. ./scripts/setup_environment.sh")
    print("3. source venv/bin/activate")
    print("4. python scripts/download_models.py")
    print("5. python -m uvicorn api.app:app --reload")
    
    print("\n📚 For the complete implementation files, see the documentation.")
    print("   The full source code for each module has been provided separately.")

if __name__ == "__main__":
    main()
