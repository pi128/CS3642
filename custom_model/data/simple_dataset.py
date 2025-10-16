# -*- coding: utf-8 -*-
"""
Simple Guitar Dataset without librosa dependency
Uses basic audio processing with soundfile and numpy.
"""

import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

class SimpleGuitarDataset(Dataset):
    """Simple dataset for guitar chord transcription training."""
    
    def __init__(self, data_dir: str, split: str = 'train', sample_rate: int = 22050, 
                 duration: float = 2.0, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        
        # Chord mapping from filename to chord name and tablature
        self.chord_mapping = {
            'a-major': ('A', {'E': 0, 'A': 0, 'D': 2, 'G': 2, 'B': 2, 'e': 0}),
            'b-major': ('B', {'E': 2, 'A': 2, 'D': 4, 'G': 4, 'B': 4, 'e': 2}),
            'c-major': ('C', {'E': 3, 'A': 3, 'D': 0, 'G': 0, 'B': 1, 'e': 0}),
            'd-major': ('D', {'E': 0, 'A': 0, 'D': 0, 'G': 2, 'B': 3, 'e': 2}),
            'e-major': ('E', {'E': 0, 'A': 2, 'D': 2, 'G': 1, 'B': 0, 'e': 0}),
            'f-major': ('F', {'E': 1, 'A': 3, 'D': 3, 'G': 2, 'B': 1, 'e': 1}),
            'g-major': ('G', {'E': 3, 'A': 2, 'D': 0, 'G': 0, 'B': 0, 'e': 3}),
        }
        
        # Load and process audio files
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_samples(self) -> List[Dict]:
        """Load and process all audio samples."""
        samples = []
        
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist")
            return samples
        
        # Find all .wav files
        audio_files = list(self.data_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            # Extract chord name from filename
            # Pattern: 17567__danglada__a-major.wav
            filename_parts = audio_file.stem.split('__')
            if len(filename_parts) >= 3:
                chord_key = filename_parts[2].lower()  # a-major
            else:
                continue
            
            if chord_key in self.chord_mapping:
                chord_name, tablature = self.chord_mapping[chord_key]
                
                try:
                    # Load audio with soundfile
                    audio, sr = sf.read(str(audio_file))
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    # Resample if needed (simple linear interpolation)
                    if sr != self.sample_rate:
                        audio = self._resample_audio(audio, sr, self.sample_rate)
                    
                    # Trim to duration
                    target_length = int(self.duration * self.sample_rate)
                    if len(audio) > target_length:
                        audio = audio[:target_length]
                    else:
                        # Pad if too short
                        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
                    
                    # Normalize
                    audio = audio / (np.max(np.abs(audio)) + 1e-8)
                    
                    samples.append({
                        'audio': audio,
                        'chord_name': chord_name,
                        'tablature': tablature,
                        'filename': audio_file.name
                    })
                    
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
                    continue
        
        # Split into train/val if needed
        if self.split == 'train':
            # Use 80% for training
            samples = samples[:int(0.8 * len(samples))]
        elif self.split == 'val':
            # Use 20% for validation
            samples = samples[int(0.8 * len(samples)):]
        
        return samples
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio
        
        # Calculate new length
        new_length = int(len(audio) * target_sr / orig_sr)
        
        # Create time indices
        orig_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, orig_indices, audio)
        
        return resampled
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        sample = self.samples[idx]
        audio = sample['audio'].copy()
        
        # Data augmentation for training
        if self.augment and self.split == 'train':
            audio = self._augment_audio(audio)
        
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        # Encode chord and tablature
        chord_target = self._encode_chord(sample['chord_name'])
        tablature_target = self._encode_tablature(sample['tablature'])
        
        return {
            'input': audio_tensor,
            'chord_target': chord_target,
            'tablature_target': tablature_target,
            'chord_name': sample['chord_name'],
            'filename': sample['filename']
        }
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation to audio."""
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, audio.shape)
            audio = audio + noise
        
        # Random volume scaling
        if random.random() < 0.5:
            scale = random.uniform(0.7, 1.3)
            audio = audio * scale
        
        # Random time shift (circular shift)
        if random.random() < 0.3:
            shift = random.randint(0, len(audio) // 4)
            audio = np.roll(audio, shift)
        
        return audio
    
    def _encode_chord(self, chord_name: str) -> torch.Tensor:
        """Encode chord name as one-hot vector."""
        chord_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G'
        ]
        
        if chord_name in chord_names:
            idx = chord_names.index(chord_name)
            target = torch.zeros(len(chord_names))
            target[idx] = 1.0
            return target
        else:
            return torch.zeros(len(chord_names))
    
    def _encode_tablature(self, tablature: Dict[str, int]) -> torch.Tensor:
        """Encode tablature as tensor."""
        # 6 strings, 25 frets (0-24)
        tab_tensor = torch.zeros(6, 25)
        
        string_names = ['E', 'A', 'D', 'G', 'B', 'e']
        
        for i, string_name in enumerate(string_names):
            if string_name in tablature and tablature[string_name] is not None:
                fret = tablature[string_name]
                if 0 <= fret <= 24:
                    tab_tensor[i, fret] = 1.0
        
        return tab_tensor
    
    def get_chord_names(self) -> List[str]:
        """Get list of chord names."""
        return ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def create_simple_dataloaders(data_dir: str, batch_size: int = 8, num_workers: int = 0) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""
    
    # Create datasets
    train_dataset = SimpleGuitarDataset(data_dir, split='train', augment=True)
    val_dataset = SimpleGuitarDataset(data_dir, split='val', augment=False)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Test the dataset
if __name__ == "__main__":
    # Test with your chord files
    data_dir = "/Users/james/Documents/CS3642/CS3642/project/test_chords"
    
    print("Testing Simple Guitar Dataset")
    print("=" * 50)
    
    # Create dataset
    dataset = SimpleGuitarDataset(data_dir, split='train')
    
    if len(dataset) > 0:
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Test a sample
        sample = dataset[0]
        print(f"Sample shapes:")
        print(f"  Input: {sample['input'].shape}")
        print(f"  Chord target: {sample['chord_target'].shape}")
        print(f"  Tablature target: {sample['tablature_target'].shape}")
        print(f"  Chord: {sample['chord_name']}")
        print(f"  File: {sample['filename']}")
        
        # Test dataloader
        train_loader, val_loader = create_simple_dataloaders(data_dir, batch_size=2)
        
        print(f"\nDataloaders created:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print(f"\nBatch shapes:")
            print(f"  Input: {batch['input'].shape}")
            print(f"  Chord targets: {batch['chord_target'].shape}")
            print(f"  Tablature targets: {batch['tablature_target'].shape}")
            break
            
    else:
        print("No samples loaded. Check your data directory.")
