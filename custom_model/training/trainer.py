"""
Custom Guitar Model Trainer
Training pipeline for the hand-rolled guitar transcription model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os
import json
from tqdm import tqdm

class GuitarDataset(Dataset):
    """Dataset for guitar transcription training."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = data_dir
        self.split = split
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load training data."""
        # This would load from your training data
        # For now, return empty list
        return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample."""
        sample = self.data[idx]
        
        return {
            'input': sample['input'],
            'chord_target': sample['chord_target'],
            'tablature_target': sample['tablature_target']
        }

class GuitarTrainer:
    """Trainer for the custom guitar model."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.chord_loss = nn.CrossEntropyLoss()
        self.tablature_loss = nn.BCEWithLogitsLoss()
        self.confidence_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        self.train_history = {
            'chord_loss': [],
            'tablature_loss': [],
            'confidence_loss': [],
            'total_loss': [],
            'chord_accuracy': [],
            'tablature_accuracy': []
        }
        
        self.val_history = {
            'chord_loss': [],
            'tablature_loss': [],
            'confidence_loss': [],
            'total_loss': [],
            'chord_accuracy': [],
            'tablature_accuracy': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {
            'chord_loss': 0.0,
            'tablature_loss': 0.0,
            'confidence_loss': 0.0,
            'total_loss': 0.0
        }
        
        epoch_accuracies = {
            'chord_accuracy': 0.0,
            'tablature_accuracy': 0.0
        }
        
        num_batches = len(dataloader)
        
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            input_data = batch['input'].to(self.device)
            chord_target = batch['chord_target'].to(self.device)
            tablature_target = batch['tablature_target'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_data)
            
            # Calculate losses
            chord_loss = self.chord_loss(outputs['chord_logits'], chord_target.argmax(dim=1))
            
            # Tablature loss (flatten for BCE)
            tablature_pred = outputs['fret_positions'].view(-1, 6 * 25)
            tablature_target_flat = tablature_target.view(-1, 6 * 25)
            tablature_loss = self.tablature_loss(tablature_pred, tablature_target_flat)
            
            # Confidence loss (dummy for now)
            confidence_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_loss = chord_loss + tablature_loss + confidence_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            epoch_losses['chord_loss'] += chord_loss.item()
            epoch_losses['tablature_loss'] += tablature_loss.item()
            epoch_losses['confidence_loss'] += confidence_loss.item()
            epoch_losses['total_loss'] += total_loss.item()
            
            # Calculate accuracies
            chord_pred = outputs['chord_logits'].argmax(dim=1)
            chord_target_idx = chord_target.argmax(dim=1)
            chord_accuracy = (chord_pred == chord_target_idx).float().mean().item()
            epoch_accuracies['chord_accuracy'] += chord_accuracy
            
            # Tablature accuracy (simplified)
            tablature_pred_binary = (torch.sigmoid(tablature_pred) > 0.5).float()
            tablature_accuracy = (tablature_pred_binary == tablature_target_flat).float().mean().item()
            epoch_accuracies['tablature_accuracy'] += tablature_accuracy
        
        # Average losses and accuracies
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        for key in epoch_accuracies:
            epoch_accuracies[key] /= num_batches
        
        return {**epoch_losses, **epoch_accuracies}
    
    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = {
            'chord_loss': 0.0,
            'tablature_loss': 0.0,
            'confidence_loss': 0.0,
            'total_loss': 0.0
        }
        
        epoch_accuracies = {
            'chord_accuracy': 0.0,
            'tablature_accuracy': 0.0
        }
        
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move to device
                input_data = batch['input'].to(self.device)
                chord_target = batch['chord_target'].to(self.device)
                tablature_target = batch['tablature_target'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_data)
                
                # Calculate losses
                chord_loss = self.chord_loss(outputs['chord_logits'], chord_target.argmax(dim=1))
                
                # Tablature loss
                tablature_pred = outputs['fret_positions'].view(-1, 6 * 25)
                tablature_target_flat = tablature_target.view(-1, 6 * 25)
                tablature_loss = self.tablature_loss(tablature_pred, tablature_target_flat)
                
                # Confidence loss
                confidence_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = chord_loss + tablature_loss + confidence_loss
                
                # Accumulate losses
                epoch_losses['chord_loss'] += chord_loss.item()
                epoch_losses['tablature_loss'] += tablature_loss.item()
                epoch_losses['confidence_loss'] += confidence_loss.item()
                epoch_losses['total_loss'] += total_loss.item()
                
                # Calculate accuracies
                chord_pred = outputs['chord_logits'].argmax(dim=1)
                chord_target_idx = chord_target.argmax(dim=1)
                chord_accuracy = (chord_pred == chord_target_idx).float().mean().item()
                epoch_accuracies['chord_accuracy'] += chord_accuracy
                
                # Tablature accuracy
                tablature_pred_binary = (torch.sigmoid(tablature_pred) > 0.5).float()
                tablature_accuracy = (tablature_pred_binary == tablature_target_flat).float().mean().item()
                epoch_accuracies['tablature_accuracy'] += tablature_accuracy
        
        # Average losses and accuracies
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        for key in epoch_accuracies:
            epoch_accuracies[key] /= num_batches
        
        return {**epoch_losses, **epoch_accuracies}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 100, save_dir: str = 'models') -> None:
        """Train the model."""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_metrics['total_loss'])
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Chord Acc: {train_metrics['chord_accuracy']:.4f}, "
                  f"Tab Acc: {train_metrics['tablature_accuracy']:.4f}")
            
            print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Chord Acc: {val_metrics['chord_accuracy']:.4f}, "
                  f"Tab Acc: {val_metrics['tablature_accuracy']:.4f}")
            
            # Save history
            for key in train_metrics:
                self.train_history[key].append(train_metrics[key])
                self.val_history[key].append(val_metrics[key])
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved! Val loss: {best_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_model(os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Save final model
        self.save_model(os.path.join(save_dir, 'final_model.pth'))
        
        # Plot training history
        self.plot_training_history(save_dir)
    
    def save_model(self, path: str) -> None:
        """Save model and training state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model and training state."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
    
    def plot_training_history(self, save_dir: str) -> None:
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.train_history['total_loss'], label='Train')
        axes[0, 0].plot(self.val_history['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(self.train_history['chord_loss'], label='Train')
        axes[0, 1].plot(self.val_history['chord_loss'], label='Validation')
        axes[0, 1].set_title('Chord Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy plots
        axes[1, 0].plot(self.train_history['chord_accuracy'], label='Train')
        axes[1, 0].plot(self.val_history['chord_accuracy'], label='Validation')
        axes[1, 0].set_title('Chord Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(self.train_history['tablature_accuracy'], label='Train')
        axes[1, 1].plot(self.val_history['tablature_accuracy'], label='Validation')
        axes[1, 1].set_title('Tablature Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    from models.guitar_model import create_model
    
    # Create model
    model = create_model()
    
    # Create trainer
    trainer = GuitarTrainer(model)
    
    # Create dummy datasets (replace with real data)
    train_dataset = GuitarDataset('data', 'train')
    val_dataset = GuitarDataset('data', 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=50)
