import regex as re
from collections import Counter, defaultdict
import multiprocessing
import pickle
from multiprocessing import Pool, cpu_count
from typing import Iterator
from cs336_basics.pretokenization_example import find_chunk_boundaries


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


"""

Resources: 
https://medium.com/@logan_16888/from-hours-to-seconds-optimising-bpe-tokeniser-training-f4234300d03e

"""
def process_chunk(args):
    input_path, start, end, special_tokens = args
    pat_special_token = "|".join(re.escape(t) for t in special_tokens)
    
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunks = re.split(pat_special_token, chunk)
    
    cnt_pretoken = {}
    for chunk in chunks:
        for match in re.finditer(PAT, chunk, re.UNICODE):
            token = match.group()
            cnt_pretoken[token] = cnt_pretoken.get(token, 0) + 1
    
    return cnt_pretoken

def parallel_pretokenize(input_path: str, special_tokens: list[str]) -> Counter:
    num_processes = multiprocessing.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes*4, "<|endoftext|>".encode("utf-8"))
        # create byte slices
        task_args = [
            (input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])            
        ]

    with Pool(num_processes) as pool:
        chunks = pool.map(process_chunk, task_args)


    cnt_pretoken_bytes = {}
    for chunk in chunks:
        for token, count in chunk.items():
            token_bytes = token.encode("utf-8")
            cnt_pretoken_bytes[token_bytes] = cnt_pretoken_bytes.get(token_bytes, 0) + count
    
    return cnt_pretoken_bytes
def get_pair_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

""""
vocab: {("l", "o", "w"): 5, ("n", "e", "w", "s", "t"): 6}
pair: ("s", "t")
"""
def merge_pair(pair, vocab):
    merged_vocab = {}
    new_symbol = pair[0] + pair[1]
    
    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(new_symbol)
                i+=2
            else:
                new_word.append(word[i])
                i+=1
        merged_vocab[tuple(new_word)] = freq
    
    
    return merged_vocab

    
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initial the vocab
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = bytes(token, "utf-8")
        
    # --- 1. Count words using streaming regex (memory-safe) ---
    # {low: 5, lower: 2, widest: 3, newest: 6}
    word_freqs = parallel_pretokenize(input_path, special_tokens)
    
    
    # 2. represent each word as a space-separated symbols
    # each unique word become a sequence of its characters
    # {low: 5} -> {("l", "o", "w"): 5}
    
    vocab_words = {}
    for token_bytes, freq in word_freqs.items():
        vocab_words[tuple(token_bytes[i:i+1] for i in range(len(token_bytes)))] = freq

    token_set = {tok for tok in vocab.values()}
        
    # 4. merge most frequent paris until vocab_size reaches to vocab_size
    num_merges = vocab_size - len(vocab)

    
    assert num_merges > 0, "vocab_size is too small"
    merges = []
    for step in range(num_merges):
        pairs = get_pair_stats(vocab_words)
        if not pairs:
            break
        
        max_freq = max(pairs.values())
        
        candidates = [p for p, c in pairs.items() if c == max_freq]
        
        best_pair = max(candidates) #lexicographically greatest
        
        # print(f"{step}: best={best_pair} freq={max_freq}")
        vocab_words = merge_pair(best_pair, vocab_words)
        merges.append(best_pair)
        token_set.add(best_pair[0] + best_pair[1])
    
    
    id2token = {id: token for id, token in enumerate(list(token_set))}
    
    return id2token, merges



class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]|None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        
        # appending special_tokens to the vocabulary if they aren’t already there
        self.special_tokens.sort(key=len, reverse=True) # match longest first
        vocab_set = set(self.vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in vocab_set:
                self.vocab[len(self.vocab)] = token_bytes
        del vocab_set
        
        self.token2idx = {v: k for k, v in vocab.items()}
        self.merges_set = set(merges)

    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]|None = None) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges 
        (in the same format that your BPE training code output) and (optionally) a list of special 
        tokens. This method should accept the following additional parameters:

        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (list[str] | None, optional): _description_. Defaults to None.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def _encode_pretoken(self, pre_token: str) -> list[int]:
        """Encode a pre-token into a sequence of token IDs using BPE."""
        pre_token = pre_token.encode("utf-8")
        if pre_token in self.token2idx:
            return [self.token2idx[pre_token]]

        # Apply BPE merges to the pre-token
        byte_arr = list(pre_token[i:i+1] for i in range(len(pre_token)))
        for token1, token2 in self.merges:
            if token1 not in byte_arr or token2 not in byte_arr:
                continue
            new_byte_arr = []
            i = 0
            while i < len(byte_arr)-1:
                if byte_arr[i] == token1 and byte_arr[i+1] == token2:
                    new_byte_arr.append(token1+token2)
                    i += 1
                else:
                    new_byte_arr.append(byte_arr[i])
                i += 1
            if i < len(byte_arr):
                new_byte_arr.append(byte_arr[i])
            byte_arr = new_byte_arr
        
        return list(self.token2idx[b] for b in byte_arr)
    
    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode a chunk of string into a sequence of token IDs using BPE."""
        ids = []
        for match in re.finditer(PAT, chunk):
            pre_token = match.group()
            ids.extend(self._encode_pretoken(pre_token))

        return ids

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        chunks = []
        if len(self.special_tokens)==0:
            chunks = [text]
        else:
            pat_special_token = "|".join(re.escape(s) for s in self.special_tokens)
            last = 0
            for match in re.finditer(pat_special_token, text):
                if match.start()>0:
                    chunks.append(text[last:match.start()])
                chunks.append(match.group())
                last = match.end()
            if last<len(text):
                chunks.append(text[last:])
        
        ids = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            if chunk_bytes in self.token2idx:
                ids.append(self.token2idx[chunk_bytes])
                continue

            ids.extend(self._encode_chunk(chunk))
        return ids
    
    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily
        yields token IDs. This is required for memory-eﬀicient tokenization of large files that
        we cannot directly load into memory.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        return b''.join([self.vocab[token_id] for token_id in tokens]).decode("utf-8", errors="replace")

if __name__ == '__main__':
    # Example usage
    # print("123")
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    input_path = "data/owt_train.txt"
    # vocab_size = 10_000
    vocab_size = 32000
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    print(f"longest token: {max(vocab.values(), key=len)}")
    # with open("tinystories.vocab", "wb") as f:
    #     pickle.dump(vocab, f)
    # with open("tinystories.merges", "wb") as f:
    with open("owt.vocab", "wb") as f:
        pickle.dump(vocab, f)
    with open("owt.merges", "wb") as f:
        pickle.dump(merges, f)