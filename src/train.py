"""
Training Pipeline for Document Understanding Model
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Dict, List
import json

from .config import load_config
from .models.multimodal_model import DocumentUnderstandingModel


class DocumentDataset(Dataset):
    """Dataset for document understanding"""
    
    def __init__(self, data_dir: str, processor, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load training samples"""
        samples = []
        
        # Load from JSON manifest
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                samples = json.load(f)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image and text
        image_path = self.data_dir / sample['image']
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs
        encoding = self.processor(
            image,
            text=sample.get('words', []),
            boxes=sample.get('boxes', []),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add labels
        encoding['labels'] = torch.tensor(sample['label'], dtype=torch.long)
        
        if 'token_labels' in sample:
            encoding['token_labels'] = torch.tensor(sample['token_labels'], dtype=torch.long)
        
        return encoding


class Trainer:
    """Model trainer"""
    
    def __init__(self, model, config, train_dataset, val_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training config
        self.batch_size = config.training.batch_size
        self.num_epochs = config.training.num_epochs
        self.learning_rate = config.training.learning_rate
        self.warmup_steps = config.training.warmup_steps
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                bbox=batch.get('bbox', torch.zeros_like(batch['input_ids']).unsqueeze(-1).repeat(1, 1, 4)),
                pixel_values=batch['pixel_values'],
                labels=batch['labels'],
                token_labels=batch.get('token_labels')
            )
            
            loss = outputs['loss']
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            preds = torch.argmax(outputs['class_logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            # Update progress
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if self.val_dataset is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    bbox=batch.get('bbox', torch.zeros_like(batch['input_ids']).unsqueeze(-1).repeat(1, 1, 4)),
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels'],
                    token_labels=batch.get('token_labels')
                )
                
                total_loss += outputs['loss'].item()
                
                preds = torch.argmax(outputs['class_logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(
            all_labels, all_preds,
            target_names=['Invoice', 'Resume', 'Report']
        ))
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self):
        """Full training loop"""
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            
            # Evaluate
            if self.val_dataset:
                val_metrics = self.evaluate()
                logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
                
                self.training_history['val_loss'].append(val_metrics['loss'])
                self.training_history['val_acc'].append(val_metrics['accuracy'])
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"Saved best model with accuracy: {self.best_val_acc:.4f}")
            
            # Save checkpoint every epoch
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        logger.info("Training completed!")
        
        # Save training history
        self.save_training_history()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        logger.info(f"Checkpoint saved: {checkpoint_dir / filename}")
    
    def save_training_history(self):
        """Save training history to JSON"""
        output_dir = Path('outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved: {output_dir / 'training_history.json'}")


def main():
    """Main training function"""
    # Load config
    config = load_config()
    
    # Create model
    model = DocumentUnderstandingModel({
        'name': config.model.name,
        'num_labels': config.model.num_labels,
        'max_length': config.model.max_length
    })
    
    # Load datasets
    logger.info("Loading datasets...")
    # Note: You need to create your datasets
    # train_dataset = DocumentDataset('data/train', model.processor)
    # val_dataset = DocumentDataset('data/val', model.processor)
    
    # For demo purposes:
    logger.warning("No dataset loaded. Please prepare your training data.")
    logger.info("Expected data structure:")
    logger.info("data/train/manifest.json - List of samples with image paths, labels, etc.")
    logger.info("data/train/<images> - Document images")
    
    # trainer = Trainer(model, config, train_dataset, val_dataset)
    # trainer.train()


if __name__ == "__main__":
    main()
