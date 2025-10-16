#!/usr/bin/env python3
"""
Simple Training Script for Custom Guitar Model
Uses basic audio processing without librosa.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add the custom_model directory to the path
sys.path.append(str(Path(__file__).parent))

from models.guitar_model import create_model
from training.trainer import GuitarTrainer
from data.simple_dataset import create_simple_dataloaders

def main():
    """Main training function."""
    print("Custom Guitar Model Training")
    print("=" * 50)
    
    # Configuration
    data_dir = "/Users/james/Documents/CS3642/CS3642/project/test_chords"
    batch_size = 4
    epochs = 20
    lr = 0.001
    save_dir = "trained_models"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Save directory: {save_dir}")
    print("=" * 50)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # Create model
        print("\nCreating model...")
        model = create_model(num_chords=7)  # A, B, C, D, E, F, G
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create dataloaders
        print("\nLoading data...")
        train_loader, val_loader = create_simple_dataloaders(
            data_dir, 
            batch_size=batch_size, 
            num_workers=0
        )
        
        if len(train_loader) == 0:
            print("Error: No training data found!")
            return
        
        print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
        
        # Create trainer
        print("\nCreating trainer...")
        trainer = GuitarTrainer(model, device=device)
        
        # Update learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr
        
        print(f"Trainer created with learning rate: {lr}")
        
        # Train model
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 50)
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            save_dir=save_dir
        )
        
        print("\n" + "=" * 50)
        print("Training completed successfully!")
        print(f"Models saved to: {save_dir}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

def quick_test():
    """Quick test to make sure everything works."""
    print("Quick Test Mode")
    print("=" * 30)
    
    # Test data loading
    data_dir = "/Users/james/Documents/CS3642/CS3642/project/test_chords"
    
    try:
        train_loader, val_loader = create_simple_dataloaders(data_dir, batch_size=2)
        
        print(f"Data loading works: {len(train_loader)} train, {len(val_loader)} val batches")
        
        # Test model creation
        model = create_model(num_chords=7)
        print(f"Model creation works: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test a forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        for batch in train_loader:
            inputs = batch['input'].to(device)
            outputs = model(inputs)
            
            print(f"Forward pass works:")
            print(f"  Input shape: {inputs.shape}")
            print(f"  Chord logits: {outputs['chord_logits'].shape}")
            print(f"  Fret positions: {outputs['fret_positions'].shape}")
            print(f"  String activations: {outputs['string_activations'].shape}")
            break
        
        print("\nAll tests passed! Ready for training.")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        main()

