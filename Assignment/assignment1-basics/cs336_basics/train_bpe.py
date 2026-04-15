import regex as re
from collections import Counter

from torch import special

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_sample_file(file_name:str,special_token:str):
    pass


def init_vocab():
    vocab = {i: bytes([i]) for i in range(256)}
    return vocab

def word_to_byte_tuple(token: str):
    return tuple(bytes([b]) for b in token.encode("utf-8"))

def count_pair(word_counts: Counter[tuple[bytes, ...]]):
    pair_counts = Counter()
    for word,freq in word_counts.items():
        if len(word) < 2:
            continue
        for pair in zip(word,word[1:]):
            pair_counts[pair] += freq
    return pair_counts
    

def apply_merge_token(word_counts,best_pair: tuple):
    merge_token = best_pair[0] + best_pair[1]
    new_word_counts = Counter()
    
    for word,freq in word_counts.items():
        new_word = []
        i = 0
        
        while i < len(word):
            if i < len(word) - 1 and (word[i],word[i + 1]) == best_pair:
                new_word.append(merge_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_counts[tuple(new_word)] += freq
    return new_word_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
        
    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} 太小，至少要容纳 256 个字节和 {len(special_tokens)} 个 special tokens。"
        )
    
    with open(input_path,"r",encoding="utf-8") as f:
        text = f.read()
    
    if special_tokens:
        escaped_tokens = [re.escape(tok) for tok in special_tokens]
        pattern = "|".join(sorted(escaped_tokens, key=len, reverse=True))
        segments = [seg for seg in re.split(pattern,text) if seg]
    else:
        segments = [text]
    
    word_counts = Counter()
    for segment in segments:
        for match in re.finditer(GPT2_PAT, segment):
            pretoken = match.group(0)
            byte_tuple = word_to_byte_tuple(pretoken)
            word_counts[byte_tuple] += 1
    
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    
    for _ in range(num_merges):
        pair_counts = count_pair(word_counts)
        if not pair_counts:
            break
    
        best_pair = max(pair_counts.items(), key=lambda x: (x[1],x[0]))[0]
        merge_token = best_pair[0] + best_pair[1]
        
        merges.append(best_pair)
        vocab[len(vocab)] = merge_token
        
        word_counts = apply_merge_token(word_counts,best_pair)
        
    return vocab,merges



