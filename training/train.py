"""
Main training script for Akia HRM - Kaggle Multi-GPU
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer
import argparse
from pathlib import Path
import os

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import AkiaConfig, AkiaForCausalLM
from training.dataset import create_dataloaders
from training.trainer import AkiaTrainer
from training.utils import (
    set_seed,
    setup_logging,
    print_model_summary,
    save_model_config
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Akia HRM")
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True, help='Path to akia_training.pkl')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    parser.add_argument('--embedding_dim', type=int, default=384)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_levels', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=6)
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    
    # Multi-GPU
    parser.add_argument('--use_ddp', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--local_rank', type=int, default=-1)
    
    # Other arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def setup_distributed():
    """Setup for distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    
    return rank, world_size, local_rank


def main():
    args = parse_args()
    
    # Setup distributed if requested
    if args.use_ddp:
        rank, world_size, local_rank = setup_distributed()
        device = f'cuda:{local_rank}'
        is_main = rank == 0
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        is_main = True
    
    # Setup (only on main process)
    if is_main:
        set_seed(args.seed)
        logger = setup_logging(args.log_dir)
        logger.info("Starting Akia HRM Training")
        logger.info(f"Arguments: {args}")
    
    # Create config
    config = AkiaConfig(
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_hierarchy_levels=args.num_levels,
        num_attention_heads=args.num_heads,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Initialize model
    if is_main:
        logger.info("Initializing model...")
    
    model = AkiaForCausalLM(config)
    
    if is_main:
        print_model_summary(model.model)
        config_path = Path(args.checkpoint_dir) / 'model_config.json'
        save_model_config(model.model, config_path)
    
    # Wrap with DDP if using multi-GPU
    if args.use_ddp:
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
    else:
        model = model.to(device)
    
    # Load tokenizer
    if is_main:
        logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloaders
    if is_main:
        logger.info("Creating dataloaders...")
    
    train_loader, val_loader = create_dataloaders(
        data_pkl=args.data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        val_split=args.val_split
    )
    
    if is_main:
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize trainer
    if is_main:
        logger.info("Initializing trainer...")
    
    trainer = AkiaTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume_from and is_main:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    if is_main:
        logger.info("Starting training...")
    
    try:
        trainer.train(num_epochs=args.num_epochs)
        
        if is_main:
            logger.info("Training completed successfully!")
            
            # Save final model
            final_path = Path(args.checkpoint_dir) / 'final_model.pt'
            torch.save({
                'model_state_dict': model.module.state_dict() if args.use_ddp else model.state_dict(),
                'config': config.to_dict()
            }, final_path)
            logger.info(f"Final model saved to {final_path}")
        
    except KeyboardInterrupt:
        if is_main:
            logger.warning("Training interrupted by user")
            emergency_path = Path(args.checkpoint_dir) / 'emergency_checkpoint.pt'
            torch.save({
                'model_state_dict': model.module.state_dict() if args.use_ddp else model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'global_step': trainer.global_step,
                'config': config.to_dict()
            }, emergency_path)
            logger.info(f"Emergency checkpoint saved to {emergency_path}")
    
    except Exception as e:
        if is_main:
            logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        if args.use_ddp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
