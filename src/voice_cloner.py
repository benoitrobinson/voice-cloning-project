"""
Test file for Voice Cloner functionality
Run with: pytest tests/test_voice_cloner.py -v
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import soundfile as sf
from unittest.mock import Mock, patch

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.voice_cloner import VoiceCloner
from src.audio_processor import AudioProcessor
from src.consent_manager import ConsentManager, ConsentType
from config.settings import Settings


class TestVoiceCloner:
    """Test suite for voice cloning functionality."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        settings = Settings()
        settings.voice_storage_path = Path(tempfile.mkdtemp())
        settings.use_gpu = False  # Use CPU for tests
        return settings
    
    @pytest.fixture
    def voice_cloner(self, settings):
        """Create voice cloner instance."""
        with patch('src.voice_cloner.TTS'):
            cloner = VoiceCloner(settings)
            # Mock the model
            cloner.model = Mock()
            cloner.model.synthesizer = Mock()
            cloner.model.synthesizer.compute_speaker_embedding = Mock(
                return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=Mock(tolist=Mock(return_value=[0.1]*256))))))
            )
            return cloner
    
    @pytest.fixture
    def sample_audio(self, settings):
        """Create sample audio for testing."""
        duration = 5  # seconds
        sample_rate = settings.sample_rate
        samples = int(duration * sample_rate)
        
        # Generate sine wave
        frequency = 440  # A4
        t = np.linspace(0, duration, samples, False)
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, audio, sample_rate)
        
        return Path(temp_file.name)
    
    def test_voice_cloner_initialization(self, voice_cloner):
        """Test voice cloner initialization."""
        assert voice_cloner is not None
        assert voice_cloner.device == "cpu"
        assert voice_cloner.model is not None
    
    def test_create_voice_clone(self, voice_cloner, sample_audio):
        """Test voice clone creation."""
        speaker_name = "Test Speaker"
        language = "en"
        
        # Create voice clone
        voice_profile = voice_cloner.create_voice_clone(
            str(sample_audio),
            speaker_name,
            language
        )
        
        # Verify profile structure
        assert voice_profile["speaker_name"] == speaker_name
        assert voice_profile["language"] == language
        assert "embedding" in voice_profile
        assert "voice_id" in voice_profile
        assert "created_at" in voice_profile
        assert voice_profile["model_version"] == "xtts_v2"
        assert voice_profile["audio_duration"] > 0
    
    def test_apply_emotion(self, voice_cloner):
        """Test emotion application to voice profile."""
        # Create mock voice profile
        voice_profile = {
            "speaker_name": "Test",
            "embedding": [0.1] * 256,
            "language": "en",
            "voice_id": "test_id"
        }
        
        # Apply happy emotion
        modified_profile = voice_cloner.apply_emotion(
            voice_profile,
            "happy",
            strength=1.5
        )
        
        # Verify modifications
        assert modified_profile["emotion"] == "happy"
        assert modified_profile["emotion_strength"] == 1.5
        assert modified_profile["embedding"] != voice_profile["embedding"]
    
    def test_save_and_load_voice_profile(self, voice_cloner, settings):
        """Test saving and loading voice profiles."""
        # Create test profile
        voice_profile = {
            "voice_id": "test_voice_123",
            "speaker_name": "Test Speaker",
            "embedding": [0.1] * 256,
            "language": "en",
            "created_at": "2025-01-20T10:00:00"
        }
        
        # Save profile
        voice_id = voice_cloner.save_voice_profile(
            voice_profile,
            "Test Speaker",
            "consent_123"
        )
        
        assert voice_id == "test_voice_123"
        
        # Load profile
        loaded_profile = voice_cloner.load_voice_profile(voice_id)
        
        # Verify loaded data
        assert loaded_profile["voice_id"] == voice_profile["voice_id"]
        assert loaded_profile["speaker_name"] == voice_profile["speaker_name"]
        assert loaded_profile["consent_id"] == "consent_123"
    
    def test_synthesize(self, voice_cloner):
        """Test speech synthesis."""
        # Mock voice profile
        voice_profile = {
            "speaker_name": "Test",
            "embedding": [0.1] * 256,
            "language": "en"
        }
        
        # Mock TTS output
        with patch.object(voice_cloner.model, 'tts_to_file') as mock_tts:
            # Call synthesize
            text = "Hello, this is a test."
            output_path = voice_cloner.synthesize(text, voice_profile)
            
            # Verify TTS was called
            mock_tts.assert_called_once()
            assert isinstance(output_path, str)
    
    def test_synthesize_with_style_params(self, voice_cloner):
        """Test synthesis with custom style parameters."""
        voice_profile = {
            "speaker_name": "Test",
            "embedding": [0.1] * 256,
            "language": "en"
        }
        
        style_params = {
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9
        }
        
        with patch.object(voice_cloner.model, 'tts_to_file') as mock_tts:
            output_path = voice_cloner.synthesize(
                "Test text",
                voice_profile,
                style_params
            )
            
            # Verify custom parameters were used
            call_kwargs = mock_tts.call_args.kwargs
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["top_k"] == 40
            assert call_kwargs["top_p"] == 0.9


class TestAudioProcessor:
    """Test suite for audio processing functionality."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create audio processor instance."""
        settings = Settings()
        return AudioProcessor(settings)
    
    @pytest.fixture
    def noisy_audio(self, settings):
        """Create noisy audio for testing."""
        duration = 3
        sample_rate = settings.sample_rate
        samples = int(duration * sample_rate)
        
        # Generate clean signal
        t = np.linspace(0, duration, samples, False)
        clean = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        # Add noise
        noise = 0.05 * np.random.randn(samples)
        noisy = clean + noise
        
        # Save to file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_file.name, noisy, sample_rate)
        
        return Path(temp_file.name)
    
    def test_process_audio(self, audio_processor, noisy_audio):
        """Test audio processing pipeline."""
        processed = audio_processor.process_audio(noisy_audio)
        
        assert isinstance(processed, np.ndarray)
        assert len(processed) > 0
        
        # Check that audio is normalized
        assert np.abs(processed).max() <= 1.0
    
    def test_validate_recording_quality(self, audio_processor, noisy_audio):
        """Test recording quality validation."""
        result = audio_processor.validate_recording_quality(noisy_audio)
        
        assert "passed" in result
        assert "metrics" in result
        assert "recommendations" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "snr" in metrics
        assert "duration" in metrics
        assert "clipping_ratio" in metrics
    
    def test_extract_features(self, audio_processor, noisy_audio):
        """Test audio feature extraction."""
        audio, sr = sf.read(noisy_audio)
        features = audio_processor.extract_features(audio, sr)
        
        assert "f0_mean" in features
        assert "f0_std" in features
        assert "mfcc_mean" in features
        assert "spectral_centroid" in features
        assert "tempo" in features


class TestConsentManager:
    """Test suite for consent management."""
    
    @pytest.fixture
    def consent_manager(self):
        """Create consent manager instance."""
        settings = Settings()
        settings.voice_storage_path = Path(tempfile.mkdtemp())
        return ConsentManager(settings)
    
    def test_create_consent(self, consent_manager):
        """Test consent creation."""
        consent_id = consent_manager.create_consent(
            speaker_name="Test User",
            speaker_email="test@example.com",
            consent_type=ConsentType.BASIC,
            use_cases=["testing", "development"],
            duration_days=30
        )
        
        assert consent_id is not None
        assert isinstance(consent_id, str)
    
    def test_verify_consent(self, consent_manager):
        """Test consent verification."""
        # Create consent
        consent_id = consent_manager.create_consent(
            speaker_name="Test User",
            speaker_email="test@example.com",
            consent_type=ConsentType.BASIC,
            use_cases=["testing"]
        )
        
        # Verify valid consent
        assert consent_manager.verify_consent(
            consent_id,
            "Test User",
            "testing"
        ) is True
        
        # Verify invalid consent
        assert consent_manager.verify_consent(
            consent_id,
            "Wrong User",
            "testing"
        ) is False
        
        # Verify unauthorized use case
        assert consent_manager.verify_consent(
            consent_id,
            "Test User",
            "commercial"
        ) is False
    
    def test_revoke_consent(self, consent_manager):
        """Test consent revocation."""
        # Create consent
        consent_id = consent_manager.create_consent(
            speaker_name="Test User",
            speaker_email="test@example.com",
            consent_type=ConsentType.BASIC,
            use_cases=["testing"]
        )
        
        # Revoke consent
        success = consent_manager.revoke_consent(
            consent_id,
            "User request",
            "Test User"
        )
        
        assert success is True
        
        # Verify consent is no longer valid
        assert consent_manager.verify_consent(
            consent_id,
            "Test User",
            "testing"
        ) is False
    
    def test_consent_encryption(self, consent_manager):
        """Test that sensitive data is encrypted."""
        email = "sensitive@example.com"
        
        consent_id = consent_manager.create_consent(
            speaker_name="Test User",
            speaker_email=email,
            consent_type=ConsentType.BASIC,
            use_cases=["testing"]
        )
        
        # Read raw file to check encryption
        consent_file = consent_manager.consent_dir / f"{consent_id}.json"
        with open(consent_file, "r") as f:
            raw_content = f.read()
        
        # Email should not appear in plain text
        assert email not in raw_content


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def system(self):
        """Create complete system setup."""
        settings = Settings()
        settings.voice_storage_path = Path(tempfile.mkdtemp())
        settings.use_gpu = False
        
        with patch('src.voice_cloner.TTS'):
            return {
                "settings": settings,
                "audio_processor": AudioProcessor(settings),
                "consent_manager": ConsentManager(settings),
                "voice_cloner": VoiceCloner(settings)
            }
    
    def test_complete_voice_cloning_workflow(self, system):
        """Test complete workflow from consent to synthesis."""
        # 1. Create consent
        consent_id = system["consent_manager"].create_consent(
            speaker_name="Integration Test",
            speaker_email="test@integration.com",
            consent_type=ConsentType.BASIC,
            use_cases=["testing"]
        )
        
        # 2. Create sample audio
        duration = 5
        sr = system["settings"].sample_rate
        audio = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, duration, duration * sr))
        
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_audio.name, audio, sr)
        
        # 3. Process audio
        processed = system["audio_processor"].process_audio(temp_audio.name)
        
        # 4. Create voice clone (mocked)
        system["voice_cloner"].model = Mock()
        system["voice_cloner"].model.synthesizer.compute_speaker_embedding = Mock(
            return_value=Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=Mock(tolist=Mock(return_value=[0.1]*256))))))
        )
        
        voice_profile = system["voice_cloner"].create_voice_clone(
            processed,
            "Integration Test",
            "en"
        )
        
        # 5. Save voice profile
        voice_id = system["voice_cloner"].save_voice_profile(
            voice_profile,
            "Integration Test",
            consent_id
        )
        
        # 6. Verify complete setup
        assert system["consent_manager"].verify_consent(
            consent_id,
            "Integration Test",
            "testing"
        )
        
        loaded_profile = system["voice_cloner"].load_voice_profile(voice_id)
        assert loaded_profile["speaker_name"] == "Integration Test"
        assert loaded_profile["consent_id"] == consent_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
