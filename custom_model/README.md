# 🎸 Guitar Tab GUI

A professional guitar tablature generation application using custom trained machine learning models.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.14)
- Virtual environment (recommended)

### Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd custom_model
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   python launch_gui.py
   ```

## 🎯 Features

### ✅ Implemented
- **Custom Model Integration**: Uses your trained guitar transcription model
- **Unified Interface**: Supports both custom and CREPE models
- **Audio Recording**: Real-time audio capture with level monitoring
- **File Import**: Load .wav, .mp3, and other audio formats
- **Batch Processing**: Process multiple files at once
- **Configuration System**: Flexible settings and presets
- **Professional GUI**: User-friendly interface with tkinter

### 🔄 In Progress
- **Real-time Transcription**: Live audio processing
- **Advanced Visualization**: Audio waveform and spectrogram display
- **Export Options**: LaTeX, PDF, MIDI export
- **Model Comparison**: Side-by-side results from different models

## 📁 Project Structure

```
custom_model/
├── models/
│   └── guitar_model.py          # Custom guitar model
├── data/
│   └── guitar_dataset.py        # Dataset handling
├── preprocessing/
│   └── audio_processor.py       # Audio preprocessing
├── training/
│   └── trainer.py               # Model training
├── trained_models/              # Model checkpoints
├── custom_model_transcriber.py  # Custom model integration
├── tab_generator.py             # Unified transcription interface
├── audio_recorder.py            # Real-time audio capture
├── guitar_tab_gui.py            # Main GUI application
├── config.py                    # Configuration management
├── launch_gui.py                # Application launcher
├── test_integration.py          # Integration tests
└── requirements.txt             # Dependencies
```

## 🎛️ Usage

### Basic Usage
1. **Launch the GUI**: `python launch_gui.py`
2. **Load Audio**: Click "Load Audio File" or use recording
3. **Select Model**: Choose custom, CREPE, or both models
4. **Transcribe**: Click "Transcribe Audio"
5. **View Results**: Check the results panel
6. **Export**: Save results in various formats

### Recording Audio
1. Click "Start Recording"
2. Play guitar into microphone
3. Click "Stop Recording"
4. Transcribe the recorded audio

### Batch Processing
1. Click "Batch Process"
2. Select multiple audio files
3. Wait for processing to complete
4. Review batch results

## ⚙️ Configuration

### Settings
- **Audio Settings**: Sample rate, channels, silence detection
- **Model Settings**: Model type, confidence thresholds
- **Processing Settings**: Feature extraction options
- **Output Settings**: Export formats and options
- **GUI Settings**: Interface preferences

### Presets
- **Default**: Balanced settings for general use
- **High Quality**: Maximum accuracy settings
- **Fast Processing**: Quick processing for real-time use
- **Real-time**: Optimized for live audio processing

## 🧪 Testing

Run integration tests:
```bash
python test_integration.py
```

This will test:
- File structure
- Configuration system
- Audio recorder components
- Custom model structure
- GUI imports
- Tab generator functionality

## 🔧 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install torch numpy soundfile PyYAML
   # Optional: pip install librosa pyaudio
   ```

2. **Audio Recording Issues**
   - Install PyAudio: `pip install pyaudio`
   - Check microphone permissions
   - Verify audio device selection

3. **Model Loading Errors**
   - Ensure model files exist in `trained_models/`
   - Check model file paths in configuration
   - Verify PyTorch installation

4. **GUI Not Starting**
   - Check tkinter installation: `python -c "import tkinter"`
   - Verify all required files exist
   - Run integration tests

### Performance Tips

- Use GPU acceleration if available (CUDA)
- Adjust chunk size for audio recording
- Use appropriate model for your use case
- Enable batch processing for multiple files

## 📊 Model Performance

### Custom Model
- **Accuracy**: High for trained chord types
- **Speed**: Fast inference (< 1 second)
- **Memory**: Low memory usage
- **Training**: Requires labeled guitar data

### CREPE Model
- **Accuracy**: Good for general audio
- **Speed**: Moderate (requires TensorFlow)
- **Memory**: Higher memory usage
- **Training**: Pre-trained, no training needed

## 🎵 Supported Formats

### Input Audio
- WAV (recommended)
- MP3
- FLAC
- M4A
- Other formats supported by librosa

### Output Formats
- Text tablature
- JSON results
- LaTeX (planned)
- PDF (planned)
- MIDI (planned)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the CS3642 coursework. See the main project documentation for license information.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Run integration tests
3. Review the configuration settings
4. Check the console output for error messages

---

**🎸 Happy transcribing!**