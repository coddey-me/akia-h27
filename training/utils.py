"""
Training utilities for Akia HRM
"""

import torch
import random
import numpy as np
import logging
from pathlib import Path


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    return total


def get_lr(optimizer):
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def format_time(seconds):
    """Format seconds to readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class AverageMeter:
    """Compute and store average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model_config(model, save_path):
    """Save model configuration"""
    config_dict = model.config.to_dict()
    
    import json
    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Model config saved to {save_path}")


def load_model_config(config_path):
    """Load model configuration"""
    import json
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    from model.config import AkiaConfig
    config = AkiaConfig(**config_dict)
    
    return config


def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*70)
    print("AKIA HRM MODEL SUMMARY")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Vocab Size: {model.config.vocab_size:,}")
    print(f"  Max Sequence Length: {model.config.max_seq_length:,}")
    print(f"  Embedding Dim: {model.config.embedding_dim}")
    print(f"  Hidden Dim: {model.config.hidden_dim}")
    print(f"  Hierarchy Levels: {model.config.num_hierarchy_levels}")
    print(f"  Attention Heads: {model.config.num_attention_heads}")
    
    print(f"\nHierarchy Configuration:")
    for i, name in enumerate(model.config.hierarchy_names):
        attn_config = model.config.attention_config[i]
        print(f"  Level {i} ({name}): {attn_config['type']} attention", end="")
        if attn_config.get('window_size'):
            print(f", window={attn_config['window_size']}")
        else:
            print()
    
    print(f"\nTraining Configuration:")
    print(f"  Batch Size: {model.config.batch_size}")
    print(f"  Learning Rate: {model.config.learning_rate}")
    print(f"  Warmup Steps: {model.config.warmup_steps}")
    print(f"  Max Steps: {model.config.max_steps}")
    
    print(f"\nParameters:")
    count_parameters(model)
    
    print("="*70 + "\n")


def gradient_stats(model):
    """Print gradient statistics"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    return total_norm
