"""
Hierarchical Reasoning Layers
Core HRM components for multi-level abstraction
"""

import torch
import torch.nn as nn
from .attention_modules import HierarchicalAttentionLayer, CrossLevelGate


class HierarchyLevel(nn.Module):
    """Single level in the hierarchy"""
    def __init__(self, d_model, num_heads, attention_config, dropout=0.1):
        super().__init__()
        
        self.attention_layer = HierarchicalAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            attention_type=attention_config["type"],
            window_size=attention_config.get("window_size"),
            dropout=dropout
        )
        
        # Level-specific projection
        self.level_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        x = self.attention_layer(x, mask)
        x = self.level_projection(x)
        return x


class HierarchicalReasoningModule(nn.Module):
    """Core HRM with bottom-up and top-down processing"""
    def __init__(self, config):
        super().__init__()
        self.num_levels = config.num_hierarchy_levels
        self.d_model = config.hidden_dim
        
        # Create hierarchy levels
        self.levels = nn.ModuleList([
            HierarchyLevel(
                d_model=self.d_model,
                num_heads=config.num_attention_heads,
                attention_config=config.attention_config[i],
                dropout=config.dropout
            ) for i in range(self.num_levels)
        ])
        
        # Cross-level gates
        self.cross_level_gates = nn.ModuleList([
            CrossLevelGate(self.d_model) 
            for _ in range(self.num_levels - 1)
        ])
        
        # Top-down refinement gates
        self.refinement_gates = nn.ModuleList([
            CrossLevelGate(self.d_model)
            for _ in range(self.num_levels - 1)
        ])
        
        # Residual bridges (skip connections across non-adjacent levels)
        self.residual_bridges = nn.ModuleDict({
            f"{i}_{j}": nn.Linear(self.d_model, self.d_model)
            for i in range(self.num_levels)
            for j in range(i + 2, self.num_levels)
        })
        
        self.final_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        # Store outputs at each level
        level_outputs = []
        
        # Bottom-up processing
        current = x
        for level_idx, level in enumerate(self.levels):
            current = level(current, mask)
            level_outputs.append(current)
            
            # Apply cross-level gate if not last level
            if level_idx < self.num_levels - 1:
                # Look ahead to next level's representation (using previous as proxy)
                if level_idx > 0:
                    current = self.cross_level_gates[level_idx](
                        level_outputs[level_idx - 1],
                        current
                    )
        
        # Top-down refinement
        refined_outputs = [level_outputs[-1]]  # Start with top level
        
        for level_idx in range(self.num_levels - 2, -1, -1):
            refined = self.refinement_gates[level_idx](
                level_outputs[level_idx],
                refined_outputs[0]
            )
            refined_outputs.insert(0, refined)
        
        # Add residual bridges for non-adjacent levels
        bridged_outputs = refined_outputs.copy()
        for i in range(self.num_levels):
            for j in range(i + 2, self.num_levels):
                bridge_key = f"{i}_{j}"
                if bridge_key in self.residual_bridges:
                    bridge = self.residual_bridges[bridge_key](refined_outputs[i])
                    bridged_outputs[j] = bridged_outputs[j] + bridge * 0.1
        
        # Combine all levels with learned weighting
        combined = torch.stack(bridged_outputs, dim=0).mean(dim=0)
        output = self.final_norm(combined)
        
        return output, bridged_outputs


class EarlyExitClassifier(nn.Module):
    """Early exit mechanism for simple queries"""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.classifier = nn.Linear(d_model, vocab_size)
        self.confidence_threshold = nn.Parameter(torch.tensor(0.9))
        
    def forward(self, x):
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean()
        return logits, confidence
    
    def should_exit(self, confidence):
        return confidence > self.confidence_threshold
