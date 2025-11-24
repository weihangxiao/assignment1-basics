"""Prepare training data by tokenizing text and saving as binary file."""

import numpy as np
import pathlib
import sys

# Add parent directory to path to import bpe
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from data.bpe import Tokenizer

def prepare_data(input_file: str, output_file: str, tokenizer: Tokenizer):
    """Tokenize text file and save as binary .dat file."""
    
    print(f"Processing {input_file}...")
    
    # Read and tokenize in chunks to handle large files
    tokens = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens.extend(tokenizer.encode(line))
    
    print(f"Tokenized to {len(tokens)} tokens")
    
    # Save as binary
    tokens_array = np.array(tokens, dtype=np.int32)
    tokens_array.tofile(output_file)
    
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent / "data"
    
    # Load your trained tokenizer
    print("Loading tokenizer...")
    # tokenizer = Tokenizer.from_files(
    #     vocab_filepath=str(data_dir / "tinystories.vocab"),
    #     merges_filepath=str(data_dir / "tinystories.merges"),
    #     special_tokens=["<|endoftext|>"]
    # )
    # print(f"Tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")
    
    # # Prepare train and validation data
    # prepare_data(
    #     str(data_dir / "TinyStoriesV2-GPT4-train.txt"),
    #     str(data_dir / "train.dat"),
    #     tokenizer
    # )
    
    # prepare_data(
    #     str(data_dir / "TinyStoriesV2-GPT4-valid.txt"),
    #     str(data_dir / "validate.dat"),
    #     tokenizer
    # )
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(data_dir / "owt.vocab"),
        merges_filepath=str(data_dir / "owt.merges"),
        special_tokens=["<|endoftext|>"]
    )
    print(f"Tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")
    
    # Prepare train and validation data
    prepare_data(
        str(data_dir / "owt_train.txt"),
        str(data_dir / "owt_train.dat"),
        tokenizer
    )
    
    prepare_data(
        str(data_dir / "owt_valid.txt"),
        str(data_dir / "owt_valid.dat"),
        tokenizer
    )
    
    print("Done!")
