"""
Trainer for Akia HRM
"""

import os
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
            T_max=max(1, total_steps - config.warmup_steps),
            eta_min=config.learning_rate * 0.1
        )
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=max(1, config.warmup_steps)
        )
        
        # Mixed precision using new API
        # GradScaler(enabled=...) avoids the deprecated constructor warnings
        self.scaler = torch.amp.GradScaler(enabled=bool(getattr(config, 'use_fp16', False)))
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_stats = []
        
    def _extract_loss(self, outputs):
        """
        Robustly extract a scalar loss tensor from model outputs.
        Accepts: dict with 'loss', tuple/list (loss, ...), or a scalar tensor.
        """
        if isinstance(outputs, dict):
            loss = outputs.get('loss') or outputs.get('total_loss') or outputs.get('loss_value')
            if loss is None:
                # try to find a tensor-like value in dict
                for v in outputs.values():
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        loss = v
                        break
        elif isinstance(outputs, (list, tuple)):
            # assume first element is logits or loss; if first is logits, we expect model to return loss as well
            # if outputs[0] is tensor with more dims, assume outputs[1] is aux_loss
            first = outputs[0]
            if isinstance(first, torch.Tensor) and first.ndim == 0:
                loss = first
            else:
                # search for tensor scalar in tuple
                loss = None
                for v in outputs:
                    if isinstance(v, torch.Tensor) and v.ndim == 0:
                        loss = v
                        break
        elif isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = None

        return loss

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
        total_loss = 0.0
        accumulation_count = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=bool(getattr(self.config, 'use_fp16', False))):
                outputs = self.model(input_ids, attention_mask, labels)
                loss = self._extract_loss(outputs)
                if loss is None:
                    raise ValueError("Model forward did not return a recognizable loss tensor. "
                                     "Ensure forward returns a scalar loss under key 'loss' or as a scalar tensor.")
                # normalize for gradient accumulation
                loss = loss / float(self.config.gradient_accumulation_steps)
            
            # Check for NaN/Inf loss (skip step if found)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARN] NaN/Inf loss detected at global_step={self.global_step}, batch_idx={batch_idx}. Skipping this step.")
                # clear grads and continue (do not update optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                continue
            
            # Backward
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_count += 1
            
            # Gradient accumulation commit
            if accumulation_count == self.config.gradient_accumulation_steps:
                # Unscale -> clip -> step -> update scaler
                if self.scaler.is_enabled():
                    # unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    try:
                        self.scaler.step(self.optimizer)
                    except Exception as e:
                        print(f"[ERROR] scaler.step failed: {e}. Skipping optimizer step.")
                        # fallback: zero grads and continue
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulation_count = 0
                        continue
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                # Learning rate scheduling
                if self.global_step < self.config.warmup_steps:
                    try:
                        self.warmup_scheduler.step()
                    except Exception:
                        pass
                else:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass
                
                # Zero grads (use set_to_none for perf)
                self.optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                self.global_step += 1
                
                # Logging (note: loss here is the last accumulation piece)
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.training_stats.append({
                        'step': self.global_step,
                        'loss': float(loss.detach().cpu().item() * self.config.gradient_accumulation_steps),
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
            
            # accumulate reporting loss value to total_loss for averaging
            total_loss += float(loss.detach().cpu().item() * self.config.gradient_accumulation_steps)
            progress_bar.set_postfix({'loss': float(loss.detach().cpu().item() * self.config.gradient_accumulation_steps)})
            
            # periodic cleanup to help memory fragmentation
            if (batch_idx + 1) % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        # Avoid division by zero if DataLoader length 0
        num_batches = max(1, len(self.train_loader))
        return total_loss / num_batches
    
    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = self._extract_loss(outputs)
                if loss is None:
                    raise ValueError("Validation forward did not return a recognizable loss tensor.")
                
                total_loss += float(loss.detach().cpu().item())
                num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _save_checkpoint(self, epoch=None, val_loss=None, step=None, is_best=False):
        checkpoint = {
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
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
