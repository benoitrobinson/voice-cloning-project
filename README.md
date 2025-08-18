# Voice Cloning System 🎤

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
