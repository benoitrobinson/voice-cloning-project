"""Configuration settings"""
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
