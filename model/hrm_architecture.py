"""
Akia HRM - Main Architecture
Complete hierarchical reasoning model
"""

import torch
import torch.nn as nn
from .config import AkiaConfig
from .hierarchical_layers import HierarchicalReasoningModule, EarlyExitClassifier


class AkiaHRM(nn.Module):
    """
    Akia Hierarchical Reasoning Model
    ~27M parameters optimized for coding tasks
    """
    def __init__(self, config=None):
        super().__init__()
        
        self.config = config if config else AkiaConfig()
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.embedding_dim
        )
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(
            self.config.max_seq_length,
            self.config.embedding_dim
        )
        
        # Project to hidden dim
        self.input_projection = nn.Linear(
            self.config.embedding_dim,
            self.config.hidden_dim
        )
        
        # Core hierarchical reasoning module
        self.hrm = HierarchicalReasoningModule(self.config)
        
        # Early exit classifiers per level
        self.early_exits = nn.ModuleList([
            EarlyExitClassifier(self.config.hidden_dim, self.config.vocab_size)
            for _ in range(self.config.num_hierarchy_levels)
        ])
        
        # Output head
        self.output_projection = nn.Linear(
            self.config.hidden_dim,
            self.config.vocab_size
        )
        
        self.dropout = nn.Dropout(self.config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, use_early_exit=False):
        B, L = input_ids.shape
        
        # Create position ids
        position_ids = torch.arange(L, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_encoding(position_ids)
        
        x = token_embeds + pos_embeds
        x = self.dropout(x)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Hierarchical reasoning
        x, level_outputs = self.hrm(x, attention_mask)
        
        # Check for early exit
        if use_early_exit and self.training:
            for level_idx, (level_out, exit_clf) in enumerate(zip(level_outputs, self.early_exits)):
                logits, confidence = exit_clf(level_out)
                if exit_clf.should_exit(confidence):
                    return logits, level_idx
        
        # Final output
        logits = self.output_projection(x)
        
        return logits, level_outputs
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_hierarchy_outputs(self, input_ids, attention_mask=None):
        """Get outputs from all hierarchy levels for analysis"""
        _, level_outputs = self.forward(input_ids, attention_mask, use_early_exit=False)
        return level_outputs


class AkiaForCausalLM(nn.Module):
    """Wrapper for causal language modeling"""
    def __init__(self, config=None):
        super().__init__()
        self.model = AkiaHRM(config)
        self.config = self.model.config
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        logits, level_outputs = self.model(input_ids, attention_mask)
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute weighted loss across hierarchy levels
            loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
            # Add auxiliary losses from each hierarchy level
            aux_losses = []
            for level_idx, level_out in enumerate(level_outputs):
                level_logits = self.model.output_projection(level_out)
                shift_level_logits = level_logits[..., :-1, :].contiguous()
                
                aux_loss = loss_fct(
                    shift_level_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1)
                )
                weighted_aux = aux_loss * self.config.level_loss_weights[level_idx]
                aux_losses.append(weighted_aux)
            
            loss = main_loss + sum(aux_losses)
        
        return {"loss": loss, "logits": logits, "level_outputs": level_outputs}
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50):
        """Simple greedy generation"""
        self.model.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits, _ = self.model(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop if EOS token
                if next_token.item() == self.config.vocab_size - 1:
                    break
        
        return input_ids
