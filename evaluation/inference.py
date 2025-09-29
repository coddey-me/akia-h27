"""
Inference utilities for Akia HRM
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class AkiaInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = device
        self.model, self.config = self._load_model(checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        from model import AkiaForCausalLM, AkiaConfig
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'config' in checkpoint:
            config = AkiaConfig(**checkpoint['config'])
        else:
            config = AkiaConfig()
        
        model = AkiaForCausalLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model, config
    
    def generate(
        self,
        prompt,
        max_length=200,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    ):
        """Generate text from prompt"""
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate
        outputs = []
        with torch.no_grad():
            for _ in range(num_return_sequences):
                output_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k
                )
                
                generated_text = self.tokenizer.decode(
                    output_ids[0],
                    skip_special_tokens=True
                )
                outputs.append(generated_text)
        
        return outputs if num_return_sequences > 1 else outputs[0]
    
    def complete_code(self, low_context, high_context, code_prefix):
        """Complete code given hierarchical context"""
        prompt = f"[LOW: {low_context}] [HIGH: {high_context}] [INPUT: {code_prefix}]"
        return self.generate(prompt, temperature=0.5)
    
    def analyze_hierarchy(self, text):
        """Analyze how different hierarchy levels process the text"""
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            level_outputs = self.model.model.get_hierarchy_outputs(input_ids)
        
        # Analyze each level
        analysis = {}
        for level_idx, level_out in enumerate(level_outputs):
            probs = torch.softmax(level_out, dim=-1)
            
            # Get top predictions at this level
            top_probs, top_indices = torch.topk(probs[0, -1], k=5)
            top_tokens = [self.tokenizer.decode([idx]) for idx in top_indices]
            
            analysis[f'level_{level_idx}'] = {
                'top_tokens': top_tokens,
                'top_probs': top_probs.cpu().tolist(),
                'entropy': -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            }
        
        return analysis
    
    def batch_generate(self, prompts, **kwargs):
        """Generate for multiple prompts"""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


def interactive_mode(checkpoint_path, device='cuda'):
    """Interactive inference mode"""
    print("Loading Akia HRM...")
    inferencer = AkiaInference(checkpoint_path, device)
    print("Model loaded! Type 'quit' to exit.\n")
    
    while True:
        print("\n" + "="*70)
        low_ctx = input("Low-level context: ")
        if low_ctx.lower() == 'quit':
            break
        
        high_ctx = input("High-level context: ")
        if high_ctx.lower() == 'quit':
            break
        
        code = input("Code prefix: ")
        if code.lower() == 'quit':
            break
        
        print("\nGenerating...")
        result = inferencer.complete_code(low_ctx, high_ctx, code)
        
        print("\n" + "-"*70)
        print("RESULT:")
        print(result)
        print("-"*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Akia HRM Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--prompt', type=str, help='Single prompt for generation')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.checkpoint, args.device)
    elif args.prompt:
        inferencer = AkiaInference(args.checkpoint, args.device)
        result = inferencer.generate(args.prompt)
        print(result)
    else:
        print("Please specify --interactive or --prompt")
