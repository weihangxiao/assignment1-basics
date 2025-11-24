"""Generate text completions from trained model."""

import torch
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from cs336_basics.model import Transformer_LM, RotaryPositionalEmbedding
from cs336_basics.utils import load_checkpoint
from data.bpe import Tokenizer


def generate(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = None,
    device: str = "cuda"
) -> str:
    """Generate text completion from prompt.
    
    Args:
        model: Trained transformer model
        tokenizer: BPE tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        device: Device to run on
        
    Returns:
        Generated text completion
    """
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, seq_len)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits = model(tokens)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature  # (1, vocab_size)
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode
    generated_tokens = tokens[0].tolist()
    return tokenizer.decode(generated_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--prompt', type=str, required=True, help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    args = parser.parse_args()
    
    # Load config
    config_path = pathlib.Path(__file__).parent / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(data_dir / "tinystories.vocab"),
        merges_filepath=str(data_dir / "tinystories.merges"),
        special_tokens=["<|endoftext|>"]
    )
    
    # Build model
    model_config = config['model'].copy()
    rope_theta = model_config.pop('rope_theta')
    d_k = model_config['d_model'] // model_config['num_heads']
    rope = RotaryPositionalEmbedding(rope_theta, d_k, model_config['context_length'])
    model = Transformer_LM(**model_config, pos_encode=rope, theta=rope_theta)
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_checkpoint(args.checkpoint, model, None)
    model = model.to(device)
    model.eval()
    
    # Generate
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    completion = generate(
        model, 
        tokenizer, 
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    print(completion)


if __name__ == '__main__':
    main()
