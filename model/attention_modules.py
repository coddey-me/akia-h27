"""
Hierarchical Attention Modules
Different attention mechanisms for each hierarchy level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SparseAttention(nn.Module):
    """Sparse attention for lower hierarchy levels"""
    def __init__(self, d_model, num_heads, window_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Create local window mask
        attn_mask = self._create_window_mask(L, self.window_size, x.device)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + attn_mask
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        return out
    
    def _create_window_mask(self, seq_len, window_size, device):
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = 0
        return mask.unsqueeze(0).unsqueeze(0)


class DenseAttention(nn.Module):
    """Full dense attention for top hierarchy level"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        
        return out


class CrossLevelGate(nn.Module):
    """Gate mechanism for controlling information flow between hierarchy levels"""
    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(self, lower_level, higher_level):
        # Concatenate and compute gate
        combined = torch.cat([lower_level, higher_level], dim=-1)
        gate = torch.sigmoid(self.gate_proj(combined))
        
        # Gated combination
        output = gate * higher_level + (1 - gate) * lower_level
        return output


class HierarchicalAttentionLayer(nn.Module):
    """Complete attention layer with hierarchy-specific configuration"""
    def __init__(self, d_model, num_heads, attention_type, window_size=None, dropout=0.1):
        super().__init__()
        
        if attention_type == "sparse":
            self.attention = SparseAttention(d_model, num_heads, window_size, dropout)
        elif attention_type == "dense":
            self.attention = DenseAttention(d_model, num_heads, dropout)
        else:  # local
            self.attention = SparseAttention(d_model, num_heads, window_size, dropout)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Attention with residual
        attn_out = self.attention(self.layer_norm1(x), mask)
        x = x + attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.layer_norm2(x))
        x = x + ffn_out
        
        return x
