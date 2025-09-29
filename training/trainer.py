"""
Trainer for Akia HRM
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import json
from pathlib import Path


class AkiaTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda',
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine learning rate scheduler
        total_steps = config.max_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - config.warmup_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_stats = []
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss = self._train_epoch()
            val_loss = self._validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)
    
    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        accumulation_count = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
                loss.backward()
            
            accumulation_count += 1
            
            # Gradient accumulation
            if accumulation_count == self.config.gradient_accumulation_steps:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                # Learning rate scheduling
                if self.global_step < self.config.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                accumulation_count = 0
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.training_stats.append({
                        'step': self.global_step,
                        'loss': loss.item() * self.config.gradient_accumulation_steps,
                        'lr': current_lr
                    })
                
                # Checkpoint saving
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(
                        epoch=None,
                        val_loss=None,
                        step=self.global_step
                    )
                
                # Max steps check
                if self.global_step >= self.config.max_steps:
                    break
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            progress_bar.set_postfix({'loss': loss.item() * self.config.gradient_accumulation_steps})
        
        return total_loss / len(self.train_loader)
    
    def _validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                total_loss += outputs['loss'].item()
        
        return total_loss / len(self.val_loader)
    
    def _save_checkpoint(self, epoch=None, val_loss=None, step=None, is_best=False):
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'training_stats': self.training_stats
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        if is_best:
            save_path = self.checkpoint_dir / 'best_model.pt'
            print(f"\n✓ Saving best model (val_loss: {val_loss:.4f})")
        elif step is not None:
            save_path = self.checkpoint_dir / f'checkpoint_step_{step}.pt'
        else:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, save_path)
        
        # Save training stats
        stats_path = self.checkpoint_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.training_stats = checkpoint.get('training_stats', [])
        
        print(f"Loaded checkpoint from step {self.global_step}")
        
        return checkpoint
