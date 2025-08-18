#!/usr/bin/env python3
"""
Voice Recording Script - Interactive voice recording with quality checks
Based on the guide's recording optimization section
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import sys
import os
from pathlib import Path
import threading
import time
from datetime import datetime
import argparse
import json
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from config.settings import Settings

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class VoiceRecorder:
    """Interactive voice recording with real-time quality monitoring."""
    
    def __init__(self, settings: Settings):
        """Initialize voice recorder."""
        self.settings = settings
        self.audio_processor = AudioProcessor(settings)
        self.recording = False
        self.audio_queue = queue.Queue()
        self.current_recording = []
        self.sample_rate = settings.sample_rate
        self.channels = 1  # Mono
        
        # Recording parameters
        self.min_duration = settings.min_audio_length
        self.max_duration = settings.max_audio_length
        
        # Quality monitoring
        self.monitor_queue = queue.Queue()
        self.quality_thread = None
        
    def list_devices(self):
        """List available audio devices."""
        print(f"\n{Colors.BOLD}Available Audio Devices:{Colors.RESET}")
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                default = " (DEFAULT)" if idx == sd.default.device[0] else ""
                print(f"{Colors.GREEN}{idx}{Colors.RESET}: {device['name']} "
                      f"[{device['max_input_channels']} channels]{default}")
    
    def select_device(self) -> Optional[int]:
        """Interactive device selection."""
        self.list_devices()
        
        while True:
            try:
                choice = input(f"\n{Colors.BLUE}Select input device number "
                             f"(or press Enter for default): {Colors.RESET}")
                
                if not choice:
                    return None
                
                device_id = int(choice)
                device_info = sd.query_devices(device_id)
                
                if device_info['max_input_channels'] > 0:
                    print(f"{Colors.GREEN}✓ Selected: {device_info['name']}{Colors.RESET}")
                    return device_id
                else:
                    print(f"{Colors.RED}✗ Not an input device{Colors.RESET}")
                    
            except (ValueError, sd.PortAudioError):
                print(f"{Colors.RED}✗ Invalid selection{Colors.RESET}")
    
    def test_levels(self, device_id: Optional[int] = None, duration: int = 5):
        """Test and adjust recording levels."""
        print(f"\n{Colors.BOLD}Testing Recording Levels...{Colors.RESET}")
        print(f"Speak normally for {duration} seconds...\n")
        
        test_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"{Colors.YELLOW}⚠ {status}{Colors.RESET}")
            test_data.append(indata.copy())
        
        with sd.InputStream(
            device=device_id,
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=callback
        ):
            # Visual level meter
            start_time = time.time()
            while time.time() - start_time < duration:
                if test_data:
                    level = np.abs(test_data[-1]).max()
                    db = 20 * np.log10(level + 1e-10)
                    
                    # Create visual meter
                    meter_width = 50
                    filled = int((level * meter_width))
                    meter = "█" * filled + "░" * (meter_width - filled)
                    
                    # Color based on level
                    if db > -3:
                        color = Colors.RED  # Too loud
                        status = "TOO LOUD!"
                    elif db > -20:
                        color = Colors.GREEN  # Good
                        status = "GOOD     "
                    else:
                        color = Colors.YELLOW  # Too quiet
                        status = "TOO QUIET"
                    
                    print(f"\r{color}{meter} {db:6.1f} dB {status}{Colors.RESET}", 
                          end='', flush=True)
                
                time.sleep(0.1)
        
        print("\n")
        
        # Analyze recording
        if test_data:
            test_audio = np.concatenate(test_data, axis=0).flatten()
            avg_level = np.abs(test_audio).mean()
            max_level = np.abs(test_audio).max()
            
            print(f"\n{Colors.BOLD}Level Analysis:{Colors.RESET}")
            print(f"Average: {20 * np.log10(avg_level + 1e-10):.1f} dB")
            print(f"Peak: {20 * np.log10(max_level + 1e-10):.1f} dB")
            
            if max_level > 0.95:
                print(f"{Colors.RED}⚠ Clipping detected! Reduce input gain.{Colors.RESET}")
            elif avg_level < 0.01:
                print(f"{Colors.YELLOW}⚠ Signal too quiet. Increase input gain or move closer to mic.{Colors.RESET}")
            else:
                print(f"{Colors.GREEN}✓ Levels look good!{Colors.RESET}")
    
    def load_script(self, script_path: Optional[str] = None) -> list[str]:
        """Load recording script."""
        if script_path and Path(script_path).exists():
            with open(script_path, 'r') as f:
                content = f.read()
                # Extract sentences (assuming they're numbered)
                sentences = []
                for line in content.split('\n'):
                    if line.strip() and line[0].isdigit():
                        # Remove numbering
                        sentence = line.split('.', 1)[1].strip() if '.' in line else line
                        sentences.append(sentence)
                return sentences
        else:
            # Use default sentences
            return [
                "The quick brown fox jumps over the lazy dog near the riverbank.",
                "She sells seashells by the seashore on sunny summer days.",
                "Technology advances rapidly in the modern digital age.",
                "Please record your voice clearly and naturally.",
                "This sample will be used to create your voice clone."
            ]
    
    def record_sentences(
        self,
        sentences: list[str],
        device_id: Optional[int] = None,
        output_dir: Path = Path("./data/recordings/raw")
    ) -> list[Path]:
        """Record sentences interactively."""
        output_dir.mkdir(parents=True, exist_ok=True)
        recorded_files = []
        
        print(f"\n{Colors.BOLD}Recording {len(sentences)} sentences{Colors.RESET}")
        print("Press SPACE to start/stop recording, 'r' to re-record, 'q' to quit\n")
        
        for idx, sentence in enumerate(sentences, 1):
            print(f"\n{Colors.BOLD}Sentence {idx}/{len(sentences)}:{Colors.RESET}")
            print(f"{Colors.BLUE}{sentence}{Colors.RESET}")
            
            while True:
                recording_data = []
                
                # Wait for space to start
                print(f"\n{Colors.YELLOW}Press SPACE to start recording...{Colors.RESET}")
                self._wait_for_key(' ')
                
                # Start recording
                print(f"{Colors.RED}● RECORDING...{Colors.RESET} (press SPACE to stop)")
                
                recording_done = threading.Event()
                
                def callback(indata, frames, time, status):
                    if status:
                        print(f"{Colors.YELLOW}⚠ {status}{Colors.RESET}")
                    recording_data.append(indata.copy())
                
                stream = sd.InputStream(
                    device=device_id,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    callback=callback
                )
                
                with stream:
                    # Monitor for space key
                    start_time = time.time()
                    while True:
                        if self._check_key(' '):
                            break
                        if time.time() - start_time > self.max_duration:
                            print(f"\n{Colors.YELLOW}⚠ Max duration reached{Colors.RESET}")
                            break
                        time.sleep(0.1)
                
                # Process recording
                audio = np.concatenate(recording_data, axis=0).flatten()
                duration = len(audio) / self.sample_rate
                
                print(f"\n{Colors.GREEN}✓ Recorded {duration:.1f} seconds{Colors.RESET}")
                
                # Quick quality check
                quality_ok = self._quick_quality_check(audio)
                
                # Play back
                print(f"\n{Colors.BLUE}Playing back...{Colors.RESET}")
                sd.play(audio, self.sample_rate)
                sd.wait()
                
                # Ask for action
                while True:
                    action = input(f"\n{Colors.BOLD}[Enter]{Colors.RESET} Accept  "
                                 f"{Colors.BOLD}[r]{Colors.RESET} Re-record  "
                                 f"{Colors.BOLD}[q]{Colors.RESET} Quit: ").lower()
                    
                    if action == '' or action == '\n':
                        # Save recording
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"sentence_{idx:03d}_{timestamp}.wav"
                        filepath = output_dir / filename
                        
                        sf.write(filepath, audio, self.sample_rate)
                        recorded_files.append(filepath)
                        
                        print(f"{Colors.GREEN}✓ Saved: {filename}{Colors.RESET}")
                        break
                        
                    elif action == 'r':
                        print(f"{Colors.YELLOW}Re-recording...{Colors.RESET}")
                        break
                        
                    elif action == 'q':
                        return recorded_files
                
                if action == '':
                    break
        
        return recorded_files
    
    def record_continuous(
        self,
        duration: int,
        device_id: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Record continuously for specified duration."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"./data/recordings/raw/continuous_{timestamp}.wav")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{Colors.BOLD}Recording for {duration} seconds...{Colors.RESET}")
        print(f"{Colors.YELLOW}Press Ctrl+C to stop early{Colors.RESET}\n")
        
        recording_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(f"{Colors.YELLOW}⚠ {status}{Colors.RESET}")
            recording_data.append(indata.copy())
            
            # Show level meter
            level = np.abs(indata).max()
            meter_width = 50
            filled = int((level * meter_width))
            meter = "█" * filled + "░" * (meter_width - filled)
            
            db = 20 * np.log10(level + 1e-10)
            color = Colors.GREEN if -20 < db < -3 else Colors.YELLOW
            
            elapsed = len(recording_data) * frames / self.sample_rate
            remaining = duration - elapsed
            
            print(f"\r{color}{meter}{Colors.RESET} "
                  f"[{elapsed:.1f}s / {duration}s]", end='', flush=True)
        
        try:
            with sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=callback
            ):
                time.sleep(duration)
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Recording stopped by user{Colors.RESET}")
        
        # Save recording
        if recording_data:
            audio = np.concatenate(recording_data, axis=0).flatten()
            sf.write(output_path, audio, self.sample_rate)
            
            actual_duration = len(audio) / self.sample_rate
            print(f"\n\n{Colors.GREEN}✓ Saved {actual_duration:.1f}s to {output_path}{Colors.RESET}")
            
            # Full quality analysis
            print(f"\n{Colors.BOLD}Analyzing recording quality...{Colors.RESET}")
            quality_report = self.audio_processor.validate_recording_quality(output_path)
            
            self._print_quality_report(quality_report)
            
            return output_path
        else:
            print(f"\n{Colors.RED}✗ No audio recorded{Colors.RESET}")
            return None
    
    def _quick_quality_check(self, audio: np.ndarray) -> bool:
        """Quick quality check for immediate feedback."""
        # Check clipping
        if np.abs(audio).max() > 0.99:
            print(f"{Colors.RED}⚠ Clipping detected!{Colors.RESET}")
            return False
        
        # Check if too quiet
        if np.abs(audio).mean() < 0.001:
            print(f"{Colors.YELLOW}⚠ Very quiet recording{Colors.RESET}")
            return False
        
        return True
    
    def _print_quality_report(self, report: dict):
        """Print formatted quality report."""
        print(f"\n{Colors.BOLD}Quality Report:{Colors.RESET}")
        
        # Overall pass/fail
        if report['passed']:
            print(f"{Colors.GREEN}✓ PASSED - Ready for voice cloning{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ FAILED - Improvements needed{Colors.RESET}")
        
        # Metrics
        print(f"\n{Colors.BOLD}Metrics:{Colors.RESET}")
        metrics = report['metrics']
        
        # SNR
        snr = metrics.get('snr', 0)
        snr_color = Colors.GREEN if snr > 30 else Colors.YELLOW if snr > 20 else Colors.RED
        print(f"  SNR: {snr_color}{snr:.1f} dB{Colors.RESET}")
        
        # Duration
        duration = metrics.get('duration', 0)
        dur_color = Colors.GREEN if duration >= 10 else Colors.YELLOW
        print(f"  Duration: {dur_color}{duration:.1f}s{Colors.RESET}")
        
        # Clipping
        clip_ratio = metrics.get('clipping_ratio', 0)
        clip_color = Colors.GREEN if clip_ratio < 0.0001 else Colors.RED
        print(f"  Clipping: {clip_color}{clip_ratio*100:.3f}%{Colors.RESET}")
        
        # Recommendations
        if report['recommendations']:
            print(f"\n{Colors.BOLD}Recommendations:{Colors.RESET}")
            for rec in report['recommendations']:
                print(f"  • {rec}")
    
    def _wait_for_key(self, key: str):
        """Wait for specific key press."""
        if sys.platform == 'win32':
            import msvcrt
            while True:
                if msvcrt.kbhit() and msvcrt.getch().decode() == key:
                    break
                time.sleep(0.1)
        else:
            import termios, tty
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                while True:
                    char = sys.stdin.read(1)
                    if char == key:
                        break
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _check_key(self, key: str) -> bool:
        """Check if key is pressed (non-blocking)."""
        if sys.platform == 'win32':
            import msvcrt
            if msvcrt.kbhit():
                return msvcrt.getch().decode() == key
        else:
            import select, termios, tty
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                if select.select([sys.stdin], [], [], 0)[0]:
                    char = sys.stdin.read(1)
                    return char == key
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        return False
    
    def merge_recordings(
        self,
        recording_files: list[Path],
        output_path: Path,
        silence_duration: float = 0.5
    ) -> Path:
        """Merge multiple recordings into one file."""
        print(f"\n{Colors.BOLD}Merging {len(recording_files)} recordings...{Colors.RESET}")
        
        # Load all recordings
        audio_segments = []
        for file in recording_files:
            audio, sr = librosa.load(file, sr=self.sample_rate, mono=True)
            audio_segments.append(audio)
        
        # Create silence
        silence = np.zeros(int(silence_duration * self.sample_rate))
        
        # Merge with silence between
        merged = []
        for i, segment in enumerate(audio_segments):
            merged.append(segment)
            if i < len(audio_segments) - 1:
                merged.append(silence)
        
        merged_audio = np.concatenate(merged)
        
        # Save merged file
        sf.write(output_path, merged_audio, self.sample_rate)
        
        duration = len(merged_audio) / self.sample_rate
        print(f"{Colors.GREEN}✓ Merged audio saved: {output_path}{Colors.RESET}")
        print(f"  Total duration: {duration:.1f}s")
        
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive voice recording for cloning"
    )
    parser.add_argument(
        "--mode",
        choices=["sentences", "continuous", "test"],
        default="sentences",
        help="Recording mode"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration for continuous recording (seconds)"
    )
    parser.add_argument(
        "--script",
        help="Path to recording script"
    )
    parser.add_argument(
        "--output",
        help="Output path for recording"
    )
    parser.add_argument(
        "--device",
        type=int,
        help="Audio device ID"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge sentence recordings into one file"
    )
    
    args = parser.parse_args()
    
    # Initialize
    settings = Settings()
    recorder = VoiceRecorder(settings)
    
    print(f"{Colors.BOLD}🎤 Voice Recording Tool{Colors.RESET}")
    print(f"Sample rate: {settings.sample_rate}Hz")
    print(f"Min duration: {settings.min_audio_length}s")
    print(f"Max duration: {settings.max_audio_length}s")
    
    # Select device if not specified
    device_id = args.device
    if device_id is None:
        device_id = recorder.select_device()
    
    if args.mode == "test":
        # Test levels only
        recorder.test_levels(device_id)
        
    elif args.mode == "continuous":
        # Continuous recording
        recorder.test_levels(device_id, duration=3)
        
        output_path = Path(args.output) if args.output else None
        recorder.record_continuous(args.duration, device_id, output_path)
        
    else:  # sentences mode
        # Load script
        sentences = recorder.load_script(args.script)
        
        # Test levels
        recorder.test_levels(device_id, duration=3)
        
        # Record sentences
        recorded_files = recorder.record_sentences(sentences, device_id)
        
        if recorded_files:
            print(f"\n{Colors.GREEN}✓ Recorded {len(recorded_files)} files{Colors.RESET}")
            
            if args.merge:
                # Merge recordings
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                merged_path = Path(f"./data/recordings/processed/merged_{timestamp}.wav")
                recorder.merge_recordings(recorded_files, merged_path)
                
                # Run quality check on merged file
                quality_report = recorder.audio_processor.validate_recording_quality(merged_path)
                recorder._print_quality_report(quality_report)


if __name__ == "__main__":
    # Handle missing dependencies gracefully
    try:
        import sounddevice as sd
    except ImportError:
        print("Please install sounddevice: pip install sounddevice")
        sys.exit(1)
    
    main()
