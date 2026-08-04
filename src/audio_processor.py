"""
Audio Processor Module - Handles audio preprocessing and quality checks
Based on the guide's recording optimization section
"""

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy import signal
import logging
from pathlib import Path
from typing import Union, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio preprocessing for voice cloning."""
    
    def __init__(self, settings):
        """Initialize audio processor with settings."""
        self.settings = settings
        self.target_sr = settings.sample_rate
        self.target_db = -20.0  # Target RMS level in dB
        
    def process_audio(
        self,
        audio_path: Union[str, Path],
        enhance: bool = True,
        trim_silence: bool = True
    ) -> np.ndarray:
        """
        Process audio file for voice cloning.
        
        Args:
            audio_path: Path to audio file
            enhance: Apply noise reduction and enhancement
            trim_silence: Remove silence from beginning/end
            
        Returns:
            Processed audio as numpy array
        """
        logger.info(f"Processing audio: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        logger.info(f"Loaded audio: {len(audio)/sr:.2f}s @ {sr}Hz")
        
        # Resample if needed
        if sr != self.target_sr:
            logger.info(f"Resampling from {sr}Hz to {self.target_sr}Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        
        # Check duration
        duration = len(audio) / sr
        if duration < self.settings.min_audio_length:
            raise ValueError(
                f"Audio too short: {duration:.1f}s < {self.settings.min_audio_length}s minimum"
            )
        if duration > self.settings.max_audio_length:
            logger.warning(f"Audio too long: {duration:.1f}s, truncating...")
            audio = audio[:int(self.settings.max_audio_length * sr)]
        
        # Apply preprocessing pipeline
        if enhance:
            audio = self._enhance_audio(audio, sr)
        
        if trim_silence:
            audio = self._trim_silence(audio, sr)
        
        # Normalize audio
        audio = self._normalize_audio(audio)
        
        # Final quality check
        quality_metrics = self._check_quality(audio, sr)
        logger.info(f"Audio quality metrics: {quality_metrics}")
        
        if quality_metrics['snr'] < 20:
            logger.warning("Low SNR detected. Consider recording in quieter environment.")
        
        return audio
    
    def _enhance_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply audio enhancement techniques."""
        # Noise reduction using stationary noise assumption
        logger.info("Applying noise reduction...")
        audio_denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            stationary=True,
            prop_decrease=0.8
        )
        
        # High-pass filter to remove low-frequency noise
        nyquist = sr // 2
        low_cutoff = 80  # Hz
        sos = signal.butter(
            5,
            low_cutoff / nyquist,
            btype='highpass',
            output='sos'
        )
        audio_filtered = signal.sosfilt(sos, audio_denoised)
        
        # De-essing to reduce sibilance
        audio_deessed = self._de_ess(audio_filtered, sr)
        
        return audio_deessed
    
    def _de_ess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Reduce sibilance in audio."""
        # Simple de-esser targeting 5-8kHz range
        nyquist = sr // 2
        ess_low = 5000 / nyquist
        ess_high = 8000 / nyquist
        
        # Design band-stop filter
        sos = signal.butter(
            4,
            [ess_low, ess_high],
            btype='bandstop',
            output='sos'
        )
        
        # Apply with reduced effect (mix with original)
        filtered = signal.sosfilt(sos, audio)
        return 0.7 * audio + 0.3 * filtered
    
    def _trim_silence(
        self,
        audio: np.ndarray,
        sr: int,
        threshold_db: float = -40
    ) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find non-silent regions
        non_silent = audio_db > threshold_db
        non_silent_indices = np.where(non_silent)[0]
        
        if len(non_silent_indices) == 0:
            return audio
        
        # Add small padding
        pad_samples = int(0.1 * sr)  # 100ms padding
        start = max(0, non_silent_indices[0] - pad_samples)
        end = min(len(audio), non_silent_indices[-1] + pad_samples)
        
        trimmed = audio[start:end]
        logger.info(f"Trimmed {(len(audio) - len(trimmed)) / sr:.2f}s of silence")
        
        return trimmed
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target RMS level."""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Calculate scaling factor
        target_rms = 10**(self.target_db / 20)
        scale = target_rms / (rms + 1e-10)
        
        # Apply scaling with limiting
        normalized = audio * scale
        normalized = np.clip(normalized, -0.99, 0.99)
        
        logger.info(f"Normalized audio: {rms_db:.1f}dB -> {self.target_db:.1f}dB")
        
        return normalized
    
    def _check_quality(self, audio: np.ndarray, sr: int) -> dict:
        """Check audio quality metrics."""
        # Signal-to-noise ratio estimation
        # Using simple method: ratio of speech to silence power
        audio_db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        speech_power = np.mean(audio_db[audio_db > -30])
        noise_power = np.mean(audio_db[audio_db < -40])
        snr = speech_power - noise_power
        
        # Check for clipping
        clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
        
        # Spectral characteristics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        
        return {
            'snr': float(snr),
            'clipping_ratio': float(clipping_ratio),
            'spectral_centroid': float(spectral_centroid),
            'duration': len(audio) / sr,
            'sample_rate': sr
        }
    
    def extract_features(self, audio: np.ndarray, sr: int) -> dict:
        """Extract acoustic features for voice cloning."""
        features = {}
        
        # Fundamental frequency (F0)
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        features['f0_mean'] = float(np.nanmean(f0))
        features['f0_std'] = float(np.nanstd(f0))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = mfccs.mean(axis=1).tolist()
        features['mfcc_std'] = mfccs.std(axis=1).tolist()
        
        # Spectral features
        features['spectral_centroid'] = float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        )
        features['spectral_rolloff'] = float(
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        )
        
        # Tempo and rhythm
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = float(tempo)
        
        return features
    
    def save_processed_audio(
        self,
        audio: np.ndarray,
        output_path: Union[str, Path],
        sr: Optional[int] = None
    ):
        """Save processed audio to file."""
        if sr is None:
            sr = self.settings.sample_rate
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as WAV with proper bit depth
        sf.write(
            output_path,
            audio,
            sr,
            subtype=f'PCM_{self.settings.bit_depth}'
        )
        
        logger.info(f"Saved processed audio to {output_path}")
    
    def validate_recording_quality(self, audio_path: Union[str, Path]) -> dict:
        """
        Validate if recording meets quality standards.
        
        Returns dict with:
            - passed: bool
            - metrics: dict of quality metrics
            - recommendations: list of improvement suggestions
        """
        audio = self.process_audio(audio_path, enhance=False, trim_silence=False)
        sr = self.settings.sample_rate
        
        metrics = self._check_quality(audio, sr)
        features = self.extract_features(audio, sr)
        
        passed = True
        recommendations = []
        
        # Check SNR
        if metrics['snr'] < 20:
            passed = False
            recommendations.append("Record in quieter environment (SNR < 20dB)")
        
        # Check clipping
        if metrics['clipping_ratio'] > 0.001:
            passed = False
            recommendations.append("Reduce recording volume to avoid clipping")
        
        # Check duration
        if metrics['duration'] < self.settings.min_audio_length:
            passed = False
            recommendations.append(
                f"Recording too short (min {self.settings.min_audio_length}s)"
            )
        
        # Check F0 stability (for voice consistency)
        if features['f0_std'] > 100:
            recommendations.append("Try to maintain more consistent pitch")
        
        return {
            'passed': passed,
            'metrics': metrics,
            'features': features,
            'recommendations': recommendations
        }
