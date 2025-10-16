#!/usr/bin/env python3
"""
Working Guitar Tab GUI
Uses scipy for audio processing instead of librosa to avoid dependency issues.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
import threading

# Add the custom_model directory to the path
sys.path.append(str(Path(__file__).parent))

# Import with error handling
try:
    from models.guitar_model import create_model
    import torch
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Model not available: {e}")
    MODEL_AVAILABLE = False

class WorkingGuitarTabGUI:
    """Working GUI for guitar tablature generation using scipy."""
    
    def __init__(self):
        """Initialize the working GUI."""
        self.root = tk.Tk()
        self.root.title("🎸 Guitar Tab Generator (Working)")
        self.root.geometry("1000x700")
        
        # State
        self.current_file = None
        self.current_audio_data = None
        self.current_sample_rate = None
        self.model = None
        
        # Setup GUI
        self._setup_gui()
        self._initialize_model()
        
    def _setup_gui(self):
        """Setup the working GUI."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎸 Guitar Tab Generator", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Status
        self.status_var = tk.StringVar(value="Ready - Using scipy for audio processing")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=(0, 10))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # File tab
        file_frame = ttk.Frame(notebook)
        notebook.add(file_frame, text="📁 File Processing")
        
        # Transcription tab
        transcribe_frame = ttk.Frame(notebook)
        notebook.add(transcribe_frame, text="🎯 Transcription")
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="📋 Results")
        
        self._setup_file_tab(file_frame)
        self._setup_transcribe_tab(transcribe_frame)
        self._setup_results_tab(results_frame)
        
    def _setup_file_tab(self, parent):
        """Setup file processing tab."""
        # File loading
        load_frame = ttk.LabelFrame(parent, text="📁 Load Audio File", padding="10")
        load_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(load_frame, text="📁 Select Audio File", 
                  command=self._load_audio_file).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_info_var = tk.StringVar(value="No file loaded")
        ttk.Label(load_frame, textvariable=self.file_info_var).pack(side=tk.LEFT)
        
        # Audio info
        info_frame = ttk.LabelFrame(parent, text="📊 Audio Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.audio_info_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, height=15)
        self.audio_info_text.pack(fill=tk.BOTH, expand=True)
        
    def _setup_transcribe_tab(self, parent):
        """Setup transcription tab."""
        # Model info
        model_frame = ttk.LabelFrame(parent, text="🤖 Model Information", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.model_status_var = tk.StringVar(value="Model status: Loading...")
        ttk.Label(model_frame, textvariable=self.model_status_var).pack()
        
        # Transcription controls
        transcribe_frame = ttk.LabelFrame(parent, text="🎯 Transcription Controls", padding="10")
        transcribe_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(transcribe_frame, text="🎸 Transcribe Audio", 
                  command=self._transcribe_audio).pack(side=tk.LEFT, padx=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(transcribe_frame, variable=self.progress_var, 
                                           mode='determinate', length=300)
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 0))
        
        # Processing info
        process_frame = ttk.LabelFrame(parent, text="⚙️ Processing Information", padding="10")
        process_frame.pack(fill=tk.BOTH, expand=True)
        
        self.process_info_text = scrolledtext.ScrolledText(process_frame, wrap=tk.WORD, height=15)
        self.process_info_text.pack(fill=tk.BOTH, expand=True)
        
    def _setup_results_tab(self, parent):
        """Setup results tab."""
        # Results display
        self.results_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD, height=25)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results info
        self.results_info_var = tk.StringVar(value="No transcription results yet")
        ttk.Label(parent, textvariable=self.results_info_var).pack(pady=(0, 10))
        
    def _initialize_model(self):
        """Initialize the guitar model."""
        if MODEL_AVAILABLE:
            try:
                # Try to load the simple model first (better labels)
                model_path = "trained_models/best_model.pth"
                if os.path.exists(model_path):
                    # Load trained model with 7 specific chord classes
                    self.model = create_model(num_chords=7)
                    checkpoint = torch.load(model_path, map_location='cpu')
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.eval()
                    self.model.chord_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                    self.model_status_var.set("✅ Simple trained model loaded successfully")
                    self._log_process("Simple model loaded with 7 specific chord classes (A-G major)")
                else:
                    # Fallback to large model
                    model_path = "trained_models_large/best_model.pth"
                    if os.path.exists(model_path):
                        self.model = create_model(num_chords=2)
                        checkpoint = torch.load(model_path, map_location='cpu')
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()
                        self.model.chord_names = ['Major', 'Minor']
                        self.model_status_var.set("✅ Large trained model loaded successfully")
                        self._log_process("Large model loaded with 2 chord classes (Major/Minor)")
                    else:
                        # Create untrained model as last resort
                        self.model = create_model(num_chords=7)
                        self.model.eval()
                        self.model.chord_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
                        self.model_status_var.set("⚠️ Untrained model created - no trained model found")
                        self._log_process("Created untrained model - results will be random")
            except Exception as e:
                self.model_status_var.set(f"❌ Model failed to load: {e}")
                self._log_process(f"Model initialization failed: {e}")
        else:
            self.model_status_var.set("❌ Model not available - missing dependencies")
            self._log_process("Model not available - PyTorch or model files missing")
    
    def _load_audio_file(self):
        """Load an audio file using scipy."""
        file_types = [
            ("WAV files", "*.wav"),
            ("Audio files", "*.wav *.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=file_types
        )
        
        if filename:
            try:
                # Load audio with scipy
                sample_rate, audio_data = wavfile.read(filename)
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Normalize to float32
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                
                self.current_file = filename
                self.current_audio_data = audio_data
                self.current_sample_rate = sample_rate
                
                # Update UI
                duration = len(audio_data) / sample_rate
                self.file_info_var.set(f"Loaded: {os.path.basename(filename)}")
                
                # Display audio information
                info_text = f"📊 Audio File Information\n"
                info_text += f"{'='*50}\n\n"
                info_text += f"File: {os.path.basename(filename)}\n"
                info_text += f"Path: {filename}\n"
                info_text += f"Size: {os.path.getsize(filename)} bytes\n"
                info_text += f"Duration: {duration:.2f} seconds\n"
                info_text += f"Sample Rate: {sample_rate} Hz\n"
                info_text += f"Channels: 1 (mono)\n"
                info_text += f"Samples: {len(audio_data)}\n"
                info_text += f"Data Type: {audio_data.dtype}\n"
                info_text += f"Min Value: {np.min(audio_data):.4f}\n"
                info_text += f"Max Value: {np.max(audio_data):.4f}\n"
                info_text += f"RMS Level: {np.sqrt(np.mean(audio_data**2)):.4f}\n"
                
                self.audio_info_text.delete(1.0, tk.END)
                self.audio_info_text.insert(1.0, info_text)
                
                self.status_var.set(f"Loaded: {os.path.basename(filename)} ({duration:.2f}s)")
                
            except Exception as e:
                messagebox.showerror("File Error", f"Failed to load audio file: {e}")
                self.status_var.set(f"Error loading file: {e}")
    
    def _transcribe_audio(self):
        """Transcribe the current audio."""
        if not self.current_audio_data is not None:
            messagebox.showwarning("No Audio", "Please load an audio file first")
            return
        
        if not self.model:
            messagebox.showerror("No Model", "Model not available")
            return
        
        # Start processing in background thread
        self.progress_var.set(0)
        thread = threading.Thread(target=self._process_audio_thread)
        thread.daemon = True
        thread.start()
    
    def _process_audio_thread(self):
        """Process audio in background thread."""
        try:
            self.root.after(0, lambda: self._log_process("Starting transcription..."))
            self.root.after(0, lambda: self.progress_var.set(20))
            
            # Prepare audio for model
            audio = self.current_audio_data
            target_length = int(2.0 * self.current_sample_rate)
            
            # Pad or truncate to 2 seconds
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            self.root.after(0, lambda: self._log_process(f"Audio prepared: {len(audio)} samples"))
            self.root.after(0, lambda: self.progress_var.set(40))
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            self.root.after(0, lambda: self._log_process("Running model inference..."))
            self.root.after(0, lambda: self.progress_var.set(60))
            
            # Run model
            with torch.no_grad():
                outputs = self.model(audio_tensor)
                
                # Get chord prediction
                chord_probs = torch.softmax(outputs['chord_logits'], dim=1)
                chord_idx = torch.argmax(chord_probs, dim=1).item()
                chord_confidence = chord_probs[0, chord_idx].item()
                
                # Use the model's chord names (could be Major/Minor or A-G depending on which model loaded)
                predicted_chord = self.model.chord_names[chord_idx]
                
                # Get tablature prediction
                fret_positions = outputs['fret_positions'].cpu().numpy()[0]
                string_activations = outputs['string_activations'].cpu().numpy()[0]
                
                # Create tablature
                tablature = {}
                string_names = ['E', 'A', 'D', 'G', 'B', 'e']
                
                for i, (fret, activation) in enumerate(zip(fret_positions, string_activations)):
                    if activation > 0.5:  # String is played
                        tablature[string_names[i]] = int(round(fret))
                    else:
                        tablature[string_names[i]] = None
            
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self._log_process("Transcription completed!"))
            
            # Display results
            self.root.after(0, lambda: self._display_results(predicted_chord, chord_confidence, tablature))
            
        except Exception as e:
            self.root.after(0, lambda: self._log_process(f"Error: {e}"))
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
        finally:
            self.root.after(0, lambda: self.progress_var.set(0))
    
    def _log_process(self, message):
        """Log processing information."""
        self.process_info_text.insert(tk.END, f"{message}\n")
        self.process_info_text.see(tk.END)
    
    def _display_results(self, chord, confidence, tablature):
        """Display transcription results."""
        results = "🎸 TRANSCRIPTION RESULTS 🎸\n"
        results += "=" * 50 + "\n\n"
        
        results += f"🎯 Predicted Chord: {chord}\n"
        results += f"📊 Confidence: {confidence:.3f}\n\n"
        
        results += "🎸 Guitar Tablature:\n"
        results += "-" * 30 + "\n"
        
        string_names = ['e', 'B', 'G', 'D', 'A', 'E']
        for string_name in string_names:
            fret = tablature.get(string_name.upper())
            if fret is not None:
                results += f"{string_name}|--{fret:2d}--\n"
            else:
                results += f"{string_name}|-----\n"
        
        results += "\n📝 Notes detected:\n"
        for string_name, fret in tablature.items():
            if fret is not None:
                results += f"  • {string_name} string: fret {fret}\n"
            else:
                results += f"  • {string_name} string: not played\n"
        
        results += f"\n🎵 Audio file: {os.path.basename(self.current_file) if self.current_file else 'Unknown'}\n"
        results += f"⏱️  Duration: {len(self.current_audio_data) / self.current_sample_rate:.2f}s\n"
        results += f"🔊 Sample rate: {self.current_sample_rate} Hz\n"
        results += f"🤖 Model: {len(self.model.chord_names)} chord classes ({', '.join(self.model.chord_names)})\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results)
        
        self.results_info_var.set(f"Transcription completed: {chord} chord (confidence: {confidence:.3f})")
        self.status_var.set("Transcription completed successfully")
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()

def main():
    """Main function."""
    print("🎸 Starting Working Guitar Tab GUI...")
    
    try:
        app = WorkingGuitarTabGUI()
        print("✅ GUI initialized successfully")
        print("🎸 Working Guitar Tab Generator is ready!")
        print("💡 Using scipy for audio processing")
        
        app.run()
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
