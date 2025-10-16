# -*- coding: utf-8 -*-
"""
Large Guitar Dataset Loader
Handles the professional SMT guitar dataset with XML annotations.
"""

import os
import torch
import numpy as np
import soundfile as sf
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

class LargeGuitarDataset(Dataset):
    """Dataset for the large professional guitar dataset."""
    
    def __init__(self, data_dir: str, split: str = 'train', sample_rate: int = 22050, 
                 duration: float = 2.5, augment: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        
        # Load and process all samples
        self.samples = self._load_all_samples()
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def _load_all_samples(self) -> List[Dict]:
        """Load all samples from the dataset."""
        samples = []
        
        # Dataset1 - Chord samples (most useful for our task)
        dataset1_path = self.data_dir / "dataset1"
        if dataset1_path.exists():
            samples.extend(self._load_dataset1_samples(dataset1_path))
        
        # Dataset2 - More samples
        dataset2_path = self.data_dir / "dataset2"
        if dataset2_path.exists():
            samples.extend(self._load_dataset2_samples(dataset2_path))
        
        # Shuffle samples
        random.shuffle(samples)
        
        # Split into train/val
        if self.split == 'train':
            # Use 80% for training
            samples = samples[:int(0.8 * len(samples))]
        elif self.split == 'val':
            # Use 20% for validation
            samples = samples[int(0.8 * len(samples)):]
        
        return samples
    
    def _load_dataset1_samples(self, dataset_path: Path) -> List[Dict]:
        """Load samples from dataset1 (chord-focused)."""
        samples = []
        
        # Look for chord directories
        for subdir in dataset_path.iterdir():
            if subdir.is_dir() and "Chords" in subdir.name:
                chord_samples = self._load_chord_directory(subdir)
                samples.extend(chord_samples)
        
        return samples
    
    def _load_chord_directory(self, chord_dir: Path) -> List[Dict]:
        """Load samples from a chord directory."""
        samples = []
        
        audio_dir = chord_dir / "audio"
        annotation_dir = chord_dir / "annotation"
        
        if not audio_dir.exists() or not annotation_dir.exists():
            return samples
        
        # Get all audio files
        audio_files = list(audio_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            # Find corresponding annotation file
            annotation_file = annotation_dir / f"{audio_file.stem}.xml"
            
            if annotation_file.exists():
                try:
                    # Parse annotation
                    chord_info = self._parse_xml_annotation(annotation_file)
                    
                    if chord_info:
                        # Load audio
                        audio, sr = sf.read(str(audio_file))
                        
                        # Convert to mono if stereo
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        
                        # Resample if needed
                        if sr != self.sample_rate:
                            audio = self._resample_audio(audio, sr, self.sample_rate)
                        
                        # Trim to duration
                        target_length = int(self.duration * self.sample_rate)
                        if len(audio) > target_length:
                            audio = audio[:target_length]
                        else:
                            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
                        
                        # Normalize
                        audio = audio / (np.max(np.abs(audio)) + 1e-8)
                        
                        samples.append({
                            'audio': audio,
                            'chord_name': chord_info['chord_name'],
                            'tablature': chord_info['tablature'],
                            'filename': audio_file.name,
                            'guitar_type': chord_dir.parent.name
                        })
                        
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
                    continue
        
        return samples
    
    def _load_dataset2_samples(self, dataset_path: Path) -> List[Dict]:
        """Load samples from dataset2."""
        samples = []
        
        audio_dir = dataset_path / "audio"
        annotation_dir = dataset_path / "annotation"
        
        if not audio_dir.exists() or not annotation_dir.exists():
            return samples
        
        # Get all audio files
        audio_files = list(audio_dir.glob("*.wav"))
        
        for audio_file in audio_files:
            annotation_file = annotation_dir / f"{audio_file.stem}.xml"
            
            if annotation_file.exists():
                try:
                    chord_info = self._parse_xml_annotation(annotation_file)
                    
                    if chord_info:
                        # Load audio
                        audio, sr = sf.read(str(audio_file))
                        
                        if len(audio.shape) > 1:
                            audio = audio.mean(axis=1)
                        
                        if sr != self.sample_rate:
                            audio = self._resample_audio(audio, sr, self.sample_rate)
                        
                        target_length = int(self.duration * self.sample_rate)
                        if len(audio) > target_length:
                            audio = audio[:target_length]
                        else:
                            audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
                        
                        audio = audio / (np.max(np.abs(audio)) + 1e-8)
                        
                        samples.append({
                            'audio': audio,
                            'chord_name': chord_info['chord_name'],
                            'tablature': chord_info['tablature'],
                            'filename': audio_file.name,
                            'guitar_type': 'dataset2'
                        })
                        
                except Exception as e:
                    print(f"Error loading {audio_file}: {e}")
                    continue
        
        return samples
    
    def _parse_xml_annotation(self, xml_file: Path) -> Optional[Dict]:
        """Parse XML annotation file to extract chord information."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract events (notes)
            events = []
            for event in root.findall('.//event'):
                pitch = int(event.find('pitch').text)
                fret = int(event.find('fretNumber').text)
                string_num = int(event.find('stringNumber').text)
                events.append({
                    'pitch': pitch,
                    'fret': fret,
                    'string': string_num
                })
            
            if not events:
                return None
            
            # Determine chord name from filename
            filename = xml_file.stem
            chord_name = self._extract_chord_name_from_filename(filename)
            
            # Create tablature
            tablature = self._create_tablature_from_events(events)
            
            return {
                'chord_name': chord_name,
                'tablature': tablature,
                'events': events
            }
            
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return None
    
    def _extract_chord_name_from_filename(self, filename: str) -> str:
        """Extract chord name from filename."""
        # Examples: "1-E1-Major 00", "2-E1-Minor 00", "3-A2-Major 00"
        
        # Check for Major/Minor patterns
        if "Major" in filename:
            return "Major"
        elif "Minor" in filename:
            return "Minor"
        
        # For dataset2 files, try to infer from the tablature
        # This is a fallback - we'll improve this later
        return "Major"  # Default to Major for now
    
    def _create_tablature_from_events(self, events: List[Dict]) -> Dict[str, int]:
        """Create tablature dictionary from events."""
        # Guitar strings: 1=low E, 2=A, 3=D, 4=G, 5=B, 6=high E
        string_names = ['E', 'A', 'D', 'G', 'B', 'e']
        tablature = {}
        
        # Initialize all strings as not played
        for string_name in string_names:
            tablature[string_name] = None
        
        # Fill in played strings
        for event in events:
            string_idx = event['string'] - 1  # Convert to 0-based index
            if 0 <= string_idx < len(string_names):
                tablature[string_names[string_idx]] = event['fret']
        
        return tablature
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using linear interpolation."""
        if orig_sr == target_sr:
            return audio
        
        new_length = int(len(audio) * target_sr / orig_sr)
        orig_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
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
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        
        # Encode chord and tablature
        chord_target = self._encode_chord(sample['chord_name'])
        tablature_target = self._encode_tablature(sample['tablature'])
        
        return {
            'input': audio_tensor,
            'chord_target': chord_target,
            'tablature_target': tablature_target,
            'chord_name': sample['chord_name'],
            'filename': sample['filename'],
            'guitar_type': sample['guitar_type']
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
        
        # Random time shift
        if random.random() < 0.3:
            shift = random.randint(0, len(audio) // 4)
            audio = np.roll(audio, shift)
        
        return audio
    
    def _encode_chord(self, chord_name: str) -> torch.Tensor:
        """Encode chord name as one-hot vector."""
        chord_names = ['Major', 'Minor']
        
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
        return ['Major', 'Minor']

def create_large_dataloaders(data_dir: str, batch_size: int = 16, num_workers: int = 0) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders for the large dataset."""
    
    # Create datasets
    train_dataset = LargeGuitarDataset(data_dir, split='train', augment=True)
    val_dataset = LargeGuitarDataset(data_dir, split='val', augment=False)
    
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
    # Test with the large dataset
    data_dir = "/Users/james/Documents/CS3642/CS3642/custom_model/trainingData"
    
    print("Testing Large Guitar Dataset")
    print("=" * 50)
    
    # Create dataset
    dataset = LargeGuitarDataset(data_dir, split='train')
    
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
        print(f"  Guitar: {sample['guitar_type']}")
        
        # Test dataloader
        train_loader, val_loader = create_large_dataloaders(data_dir, batch_size=4)
        
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
