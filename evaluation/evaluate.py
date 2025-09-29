"""
Evaluation script for Akia HRM
"""

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
from pathlib import Path

from model import AkiaForCausalLM, AkiaConfig
from training.dataset import AkiaHRMDataset
from evaluation.metrics import calculate_perplexity, calculate_bleu, calculate_accuracy


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    if 'config' in checkpoint:
        config = AkiaConfig(**checkpoint['config'])
    else:
        config = AkiaConfig()
    
    # Initialize and load model
    model = AkiaForCausalLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test set"""
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Calculate metrics
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Count tokens (excluding padding)
            total_tokens += (labels != -100).sum().item()
    
    # Calculate final metrics
    avg_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'total_tokens': total_tokens
    }


def evaluate_by_difficulty(model, test_dataset, tokenizer, device='cuda'):
    """Evaluate model performance by task difficulty"""
    difficulties = ['small', 'medium', 'large']
    results = {}
    
    for difficulty in difficulties:
        # Filter dataset by difficulty
        filtered_data = [
            item for item in test_dataset.data 
            if item.get('difficulty') == difficulty
        ]
        
        if not filtered_data:
            continue
        
        # Create temporary dataset
        temp_dataset = type(test_dataset)(filtered_data, tokenizer)
        temp_loader = torch.utils.data.DataLoader(
            temp_dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Evaluate
        metrics = evaluate_model(model, temp_loader, device)
        results[difficulty] = metrics
        
        print(f"\n{difficulty.upper()} difficulty:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")
    
    return results


def generate_samples(model, tokenizer, prompts, max_length=200, device='cuda'):
    """Generate sample outputs"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
            
            output_ids = model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_k=50
            )
            
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            results.append({
                'prompt': prompt,
                'generated': generated_text
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Akia HRM")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--test_data', type=str, required=True, help='Test data pkl file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--output_file', type=str, default='evaluation_results.txt', help='Output file')
    parser.add_argument('--generate_samples', action='store_true', help='Generate sample outputs')
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, config = load_model(args.checkpoint, args.device)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading test data...")
    test_dataset = AkiaHRMDataset(args.test_data, tokenizer)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Overall evaluation
    print("Evaluating overall performance...")
    overall_results = evaluate_model(model, test_loader, args.device)
    
    print("\nOVERALL RESULTS:")
    print(f"  Loss: {overall_results['loss']:.4f}")
    print(f"  Perplexity: {overall_results['perplexity']:.2f}")
    print(f"  Total tokens: {overall_results['total_tokens']:,}")
    
    # Evaluation by difficulty
    print("\n" + "="*50)
    print("Evaluating by difficulty...")
    difficulty_results = evaluate_by_difficulty(model, test_dataset, tokenizer, args.device)
    
    # Generate samples if requested
    if args.generate_samples:
        print("\n" + "="*50)
        print("Generating samples...")
        
        sample_prompts = [
            "[LOW: Python sorting algorithm] [HIGH: Implementation] [INPUT: def sort_list(arr):",
            "[LOW: GitHub API documentation] [HIGH: REST endpoints] [INPUT: GET /repos",
            "[LOW: React component] [HIGH: User interface] [INPUT: function Button(props) {"
        ]
        
        samples = generate_samples(model, tokenizer, sample_prompts, device=args.device)
        
        print("\nSAMPLE GENERATIONS:")
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Prompt: {sample['prompt']}")
            print(f"Generated: {sample['generated']}")
    
    # Save results
    with open(args.output_file, 'w') as f:
        f.write("AKIA HRM EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL RESULTS:\n")
        f.write(f"  Loss: {overall_results['loss']:.4f}\n")
        f.write(f"  Perplexity: {overall_results['perplexity']:.2f}\n")
        f.write(f"  Total tokens: {overall_results['total_tokens']:,}\n\n")
        
        f.write("RESULTS BY DIFFICULTY:\n")
        for difficulty, metrics in difficulty_results.items():
            f.write(f"\n{difficulty.upper()}:\n")
            f.write(f"  Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Perplexity: {metrics['perplexity']:.2f}\n")
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
