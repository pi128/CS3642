"""
Configuration System
Manages application settings, model parameters, and user preferences.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field

@dataclass
class AudioSettings:
    """Audio processing settings."""
    sample_rate: int = 22050
    channels: int = 1
    chunk_size: int = 1024
    format: str = "float32"
    silence_threshold: float = 0.01
    max_silence_duration: float = 2.0
    auto_gain: bool = True
    normalize: bool = True

@dataclass
class ModelSettings:
    """Model configuration settings."""
    model_type: str = "custom"  # "custom", "crepe", "both"
    custom_model_path: Optional[str] = None
    device: str = "auto"  # "auto", "cpu", "cuda"
    confidence_threshold: float = 0.5
    batch_size: int = 1
    max_audio_length: float = 10.0  # seconds
    enable_gpu: bool = True

@dataclass
class ProcessingSettings:
    """Audio processing parameters."""
    preprocess_audio: bool = True
    extract_features: bool = True
    harmonic_percussive_separation: bool = True
    onset_detection: bool = True
    chord_detection: bool = True
    pitch_tracking: bool = True
    tablature_mapping: bool = True

@dataclass
class OutputSettings:
    """Output format and file settings."""
    output_formats: List[str] = field(default_factory=lambda: ["text", "json"])
    output_directory: str = "./output"
    include_visualizations: bool = True
    include_confidence_scores: bool = True
    include_timing_info: bool = True
    latex_export: bool = False
    pdf_export: bool = False
    midi_export: bool = False

@dataclass
class GUISettings:
    """GUI interface settings."""
    window_size: tuple = (1200, 800)
    theme: str = "default"  # "default", "dark", "light"
    show_levels: bool = True
    show_confidence: bool = True
    auto_save: bool = True
    auto_transcribe: bool = False
    real_time_feedback: bool = True
    font_size: int = 12
    icon_size: int = 24

@dataclass
class AppConfig:
    """Main application configuration."""
    audio: AudioSettings = field(default_factory=AudioSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    gui: GUISettings = field(default_factory=GUISettings)
    
    # Application metadata
    version: str = "1.0.0"
    last_updated: str = ""
    config_file: str = "config.json"

class ConfigManager:
    """Manages application configuration and settings."""
    
    def __init__(self, config_dir: str = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".guitar_tab_app"
        self.config_dir.mkdir(exist_ok=True)
        
        self.config_file = self.config_dir / "config.json"
        self.config = AppConfig()
        
        # Load existing configuration
        self.load_config()
    
    def load_config(self, config_file: str = None) -> bool:
        """Load configuration from file."""
        config_path = Path(config_file) if config_file else self.config_file
        
        if not config_path.exists():
            print(f"📝 No config file found at {config_path}. Using defaults.")
            return False
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration with loaded data
            self._update_config_from_dict(config_data)
            print(f"✅ Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return False
    
    def save_config(self, config_file: str = None) -> bool:
        """Save configuration to file."""
        config_path = Path(config_file) if config_file else self.config_file
        
        try:
            # Convert config to dictionary
            config_dict = asdict(self.config)
            
            # Add metadata
            from datetime import datetime
            config_dict['last_updated'] = datetime.now().isoformat()
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            print(f"✅ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {e}")
            return False
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        # Update audio settings
        if 'audio' in config_dict:
            audio_dict = config_dict['audio']
            for key, value in audio_dict.items():
                if hasattr(self.config.audio, key):
                    setattr(self.config.audio, key, value)
        
        # Update model settings
        if 'model' in config_dict:
            model_dict = config_dict['model']
            for key, value in model_dict.items():
                if hasattr(self.config.model, key):
                    setattr(self.config.model, key, value)
        
        # Update processing settings
        if 'processing' in config_dict:
            processing_dict = config_dict['processing']
            for key, value in processing_dict.items():
                if hasattr(self.config.processing, key):
                    setattr(self.config.processing, key, value)
        
        # Update output settings
        if 'output' in config_dict:
            output_dict = config_dict['output']
            for key, value in output_dict.items():
                if hasattr(self.config.output, key):
                    setattr(self.config.output, key, value)
        
        # Update GUI settings
        if 'gui' in config_dict:
            gui_dict = config_dict['gui']
            for key, value in gui_dict.items():
                if hasattr(self.config.gui, key):
                    setattr(self.config.gui, key, value)
    
    def get_audio_settings(self) -> AudioSettings:
        """Get audio settings."""
        return self.config.audio
    
    def get_model_settings(self) -> ModelSettings:
        """Get model settings."""
        return self.config.model
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get processing settings."""
        return self.config.processing
    
    def get_output_settings(self) -> OutputSettings:
        """Get output settings."""
        return self.config.output
    
    def get_gui_settings(self) -> GUISettings:
        """Get GUI settings."""
        return self.config.gui
    
    def update_audio_settings(self, **kwargs):
        """Update audio settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.audio, key):
                setattr(self.config.audio, key, value)
    
    def update_model_settings(self, **kwargs):
        """Update model settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.model, key):
                setattr(self.config.model, key, value)
    
    def update_processing_settings(self, **kwargs):
        """Update processing settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.processing, key):
                setattr(self.config.processing, key, value)
    
    def update_output_settings(self, **kwargs):
        """Update output settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.output, key):
                setattr(self.config.output, key, value)
    
    def update_gui_settings(self, **kwargs):
        """Update GUI settings."""
        for key, value in kwargs.items():
            if hasattr(self.config.gui, key):
                setattr(self.config.gui, key, value)
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = AppConfig()
        print("🔄 Configuration reset to defaults")
    
    def export_config(self, filename: str):
        """Export configuration to file."""
        try:
            config_dict = asdict(self.config)
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"✅ Configuration exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error exporting configuration: {e}")
            return False
    
    def import_config(self, filename: str):
        """Import configuration from file."""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            self._update_config_from_dict(config_dict)
            print(f"✅ Configuration imported from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error importing configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            'audio': asdict(self.config.audio),
            'model': asdict(self.config.model),
            'processing': asdict(self.config.processing),
            'output': asdict(self.config.output),
            'gui': asdict(self.config.gui),
            'version': self.config.version,
            'config_file': str(self.config_file),
            'last_updated': self.config.last_updated
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate audio settings
        if self.config.audio.sample_rate <= 0:
            issues.append("Invalid sample rate")
        if self.config.audio.channels <= 0:
            issues.append("Invalid channel count")
        if self.config.audio.chunk_size <= 0:
            issues.append("Invalid chunk size")
        
        # Validate model settings
        if self.config.model.model_type not in ['custom', 'crepe', 'both']:
            issues.append("Invalid model type")
        if self.config.model.confidence_threshold < 0 or self.config.model.confidence_threshold > 1:
            issues.append("Confidence threshold must be between 0 and 1")
        
        # Validate output settings
        if not self.config.output.output_directory:
            issues.append("Output directory not specified")
        if not self.config.output.output_formats:
            issues.append("No output formats specified")
        
        return issues
    
    def get_recommended_settings(self, audio_file: str = None) -> Dict[str, Any]:
        """Get recommended settings based on context."""
        recommendations = {
            'audio': {
                'sample_rate': 22050,
                'channels': 1,
                'silence_threshold': 0.01
            },
            'model': {
                'model_type': 'custom',
                'confidence_threshold': 0.6
            },
            'processing': {
                'preprocess_audio': True,
                'harmonic_percussive_separation': True
            }
        }
        
        # Analyze audio file if provided
        if audio_file and os.path.exists(audio_file):
            try:
                import librosa
                audio, sr = librosa.load(audio_file, sr=None)
                
                # Recommend settings based on audio properties
                if sr != 22050:
                    recommendations['audio']['sample_rate'] = sr
                
                # Adjust silence threshold based on audio level
                rms = np.sqrt(np.mean(audio ** 2))
                if rms < 0.1:
                    recommendations['audio']['silence_threshold'] = 0.005
                elif rms > 0.5:
                    recommendations['audio']['silence_threshold'] = 0.02
                
            except Exception as e:
                print(f"⚠️  Could not analyze audio file: {e}")
        
        return recommendations

# Configuration presets
class ConfigPresets:
    """Predefined configuration presets for different use cases."""
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """Get a configuration preset."""
        presets = {
            'default': {
                'audio': AudioSettings(),
                'model': ModelSettings(),
                'processing': ProcessingSettings(),
                'output': OutputSettings(),
                'gui': GUISettings()
            },
            
            'high_quality': {
                'audio': AudioSettings(
                    sample_rate=44100,
                    silence_threshold=0.005,
                    max_silence_duration=1.0
                ),
                'model': ModelSettings(
                    confidence_threshold=0.7,
                    max_audio_length=30.0
                ),
                'processing': ProcessingSettings(
                    preprocess_audio=True,
                    harmonic_percussive_separation=True
                ),
                'output': OutputSettings(
                    output_formats=['text', 'json', 'latex'],
                    include_visualizations=True
                ),
                'gui': GUISettings(
                    real_time_feedback=True,
                    show_confidence=True
                )
            },
            
            'fast_processing': {
                'audio': AudioSettings(
                    sample_rate=16000,
                    chunk_size=512,
                    silence_threshold=0.02
                ),
                'model': ModelSettings(
                    confidence_threshold=0.4,
                    max_audio_length=5.0
                ),
                'processing': ProcessingSettings(
                    preprocess_audio=False,
                    harmonic_percussive_separation=False
                ),
                'output': OutputSettings(
                    output_formats=['text'],
                    include_visualizations=False
                ),
                'gui': GUISettings(
                    auto_transcribe=True,
                    real_time_feedback=False
                )
            },
            
            'real_time': {
                'audio': AudioSettings(
                    sample_rate=22050,
                    chunk_size=1024,
                    silence_threshold=0.01,
                    max_silence_duration=0.5
                ),
                'model': ModelSettings(
                    model_type='custom',
                    confidence_threshold=0.5,
                    max_audio_length=2.0
                ),
                'processing': ProcessingSettings(
                    preprocess_audio=True,
                    extract_features=True
                ),
                'output': OutputSettings(
                    output_formats=['text'],
                    include_visualizations=False
                ),
                'gui': GUISettings(
                    real_time_feedback=True,
                    auto_transcribe=True,
                    show_levels=True
                )
            }
        }
        
        return presets.get(preset_name, presets['default'])
    
    @staticmethod
    def apply_preset(config_manager: ConfigManager, preset_name: str):
        """Apply a preset to the configuration manager."""
        preset = ConfigPresets.get_preset(preset_name)
        
        # Update configuration with preset values
        config_manager.config.audio = preset['audio']
        config_manager.config.model = preset['model']
        config_manager.config.processing = preset['processing']
        config_manager.config.output = preset['output']
        config_manager.config.gui = preset['gui']
        
        print(f"✅ Applied preset: {preset_name}")

# Example usage and testing
if __name__ == "__main__":
    print("⚙️  Testing Configuration System")
    print("=" * 50)
    
    # Create configuration manager
    config_manager = ConfigManager()
    
    # Get current configuration
    print("📋 Current configuration:")
    summary = config_manager.get_config_summary()
    for section, settings in summary.items():
        if isinstance(settings, dict):
            print(f"\n{section.upper()}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
    
    # Validate configuration
    issues = config_manager.validate_config()
    if issues:
        print(f"\n⚠️  Configuration issues: {issues}")
    else:
        print("\n✅ Configuration is valid")
    
    # Test preset application
    print(f"\n🎯 Testing presets...")
    presets_to_test = ['high_quality', 'fast_processing', 'real_time']
    
    for preset_name in presets_to_test:
        print(f"\nApplying preset: {preset_name}")
        ConfigPresets.apply_preset(config_manager, preset_name)
        
        # Show some key settings
        audio_settings = config_manager.get_audio_settings()
        model_settings = config_manager.get_model_settings()
        
        print(f"  Audio: {audio_settings.sample_rate}Hz, {audio_settings.channels}ch")
        print(f"  Model: {model_settings.model_type}, conf={model_settings.confidence_threshold}")
    
    # Reset to defaults
    print(f"\n🔄 Resetting to defaults...")
    config_manager.reset_to_defaults()
    
    # Save configuration
    print(f"\n💾 Saving configuration...")
    config_manager.save_config()
    
    print(f"\n✅ Configuration system test completed")
