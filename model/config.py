"""
Akia HRM Configuration
Architecture specifications and hyperparameters
"""

class AkiaConfig:
    # Model Architecture
    vocab_size = 50257  # GPT-2 tokenizer
    max_seq_length = 2048
    embedding_dim = 384
    hidden_dim = 512
    num_hierarchy_levels = 4
    num_attention_heads = 6
    ffn_multiplier = 2
    dropout = 0.1
    
    # Hierarchical Layer Names
    hierarchy_names = [
        "token_level",      # Level 0: Syntax, patterns
        "phrase_level",     # Level 1: Local semantics
        "document_level",   # Level 2: Global structure
        "abstract_level"    # Level 3: High-level reasoning
    ]
    
    # Attention Configuration per Level
    attention_config = {
        0: {"type": "sparse", "window_size": 64},
        1: {"type": "sparse", "window_size": 128},
        2: {"type": "local", "window_size": 256},
        3: {"type": "dense", "window_size": None}
    }
    
    # Training Configuration
    batch_size = 16
    gradient_accumulation_steps = 2
    learning_rate = 5e-4
    warmup_steps = 500
    max_steps = 10000
    weight_decay = 0.01
    max_grad_norm = 1.0
    
    # Loss weighting per hierarchy level (higher levels weighted more)
    level_loss_weights = [0.1, 0.2, 0.3, 0.4]
    
    # Checkpointing
    save_steps = 500
    eval_steps = 250
    logging_steps = 50
    
    # Optimization
    use_fp16 = True
    early_exit_threshold = 0.9  # Confidence threshold for early exit
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
