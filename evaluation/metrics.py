"""
Evaluation metrics for Akia HRM
"""

import torch
import numpy as np
from collections import Counter


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    return torch.exp(torch.tensor(loss)).item()


def calculate_accuracy(predictions, targets, ignore_index=-100):
    """Calculate token-level accuracy"""
    mask = targets != ignore_index
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy


def calculate_bleu(predictions, references, n=4):
    """Calculate BLEU score"""
    def get_ngrams(tokens, n):
        return Counter(zip(*[tokens[i:] for i in range(n)]))
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.tolist() if isinstance(pred, torch.Tensor) else pred
        ref_tokens = ref.tolist() if isinstance(ref, torch.Tensor) else ref
        
        # Remove padding and special tokens
        pred_tokens = [t for t in pred_tokens if t not in [-100, 0]]
        ref_tokens = [t for t in ref_tokens if t not in [-100, 0]]
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        
        # Calculate precision for each n-gram
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = get_ngrams(pred_tokens, i)
            ref_ngrams = get_ngrams(ref_tokens, i)
            
            if len(pred_ngrams) == 0:
                precisions.append(0)
                continue
            
            matches = sum((pred_ngrams & ref_ngrams).values())
            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            score = bp * np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            score = 0.0
        
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def calculate_rouge_l(predictions, references):
    """Calculate ROUGE-L score"""
    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.tolist() if isinstance(pred, torch.Tensor) else pred
        ref_tokens = ref.tolist() if isinstance(ref, torch.Tensor) else ref
        
        # Remove padding and special tokens
        pred_tokens = [t for t in pred_tokens if t not in [-100, 0]]
        ref_tokens = [t for t in ref_tokens if t not in [-100, 0]]
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            continue
        
        lcs_len = lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        scores.append(f1)
    
    return np.mean(scores) if scores else 0.0


def calculate_hierarchy_stats(level_outputs, labels):
    """Calculate statistics for each hierarchy level"""
    stats = {}
    
    for level_idx, level_out in enumerate(level_outputs):
        # Calculate entropy
        probs = torch.softmax(level_out, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        
        # Calculate confidence
        confidence = probs.max(dim=-1)[0].mean().item()
        
        # Calculate accuracy at this level
        predictions = level_out.argmax(dim=-1)
        mask = labels != -100
        correct = (predictions == labels) & mask
        accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0
        
        stats[f'level_{level_idx}'] = {
            'entropy': entropy,
            'confidence': confidence,
            'accuracy': accuracy
        }
    
    return stats


def calculate_code_metrics(generated_code, reference_code):
    """Calculate code-specific metrics"""
    metrics = {}
    
    # Exact match
    metrics['exact_match'] = int(generated_code.strip() == reference_code.strip())
    
    # Character-level similarity
    from difflib import SequenceMatcher
    metrics['char_similarity'] = SequenceMatcher(
        None, 
        generated_code, 
        reference_code
    ).ratio()
    
    # Line-level similarity
    gen_lines = set(generated_code.strip().split('\n'))
    ref_lines = set(reference_code.strip().split('\n'))
    
    if len(ref_lines) > 0:
        metrics['line_overlap'] = len(gen_lines & ref_lines) / len(ref_lines)
    else:
        metrics['line_overlap'] = 0.0
    
    return metrics
