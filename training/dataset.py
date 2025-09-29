"""
Dataset loader for Akia HRM training
"""

import torch
from torch.utils.data import Dataset, random_split
import pickle


class AkiaHRMDataset(Dataset):
    """Dataset for hierarchical reasoning training"""
    
    def __init__(self, pkl_file, tokenizer, max_length=2048):
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        low_ctx = item.get('low_level_context', '')
        high_ctx = item.get('high_level_context', '')
        input_seq = item.get('input_sequence', '')
        target = item.get('target_sequence', '')
        
        full_input = f"[LOW: {low_ctx}] [HIGH: {high_ctx}] [INPUT: {input_seq}] [TARGET: {target}]"
        
        encoding = self.tokenizer(
            full_input,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    # Remove metadata entirely

def create_dataloaders(data_pkl, tokenizer, batch_size=16, val_split=0.1, num_workers=2):
    """Create train and validation dataloaders from single dataset"""
    
    # Load full dataset
    full_dataset = AkiaHRMDataset(data_pkl, tokenizer)
    
    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
