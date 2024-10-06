"""
Byte pair encoding utilities
"""

import os
import json
import regex as re
from functools import lru_cache

@lru_cache()
def bytes_to_unicode():
    """
    The function converts a range of UTF-8 byte values (from 0 to 255) into Unicode characters,
    ensuring that there is a one-to-one, reversible mapping. The output is a dictionary where:
    - Keys are the byte values (0-255).
    - Values are the corresponding Unicode characters.

    For example, a byte value 33 (which is "!") would map to the Unicode character "!",
    while byte values that are not part of the initial range will map to unique Unicode
    characters starting from 256 onward.
    """
    # bs is a list of selected byte values that correspond to printable ASCII and Latin-1 characters
    printable_ascii_characters = list(range(ord("!"), ord("~")+1))
    extended_ascii = list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    bs = printable_ascii_characters + extended_ascii
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word: tuple):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).

    Example output:
    ('w', 'o'), ('o', 'r'), ('r', 'd')
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        # Dictionary that holds the ranking of byte pairs (bigrams) based
        # on their frequency in the corpus. More frequent pairs have lower ranks.
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token: str):
        """
        Implements Byte Pair Encoding (BPE), which is a subword tokenization technique.
        The function splits a word (given as token) into subword units based on the
        frequency of byte pairs. This helps in encoding out-of-vocabulary words by
        breaking them into known subword units.

        Example:
            - Let's say token = "word" and the self.bpe_ranks indicates that merging ('o', 'r') is frequent.
            - word = ("w", "o", "r", "d")
            - pairs = ('w', 'o'), ('o', 'r'), ('r', 'd')
            - The most frequent bigram is ('o', 'r'), so it merges into ("w", "or", "d")
            - The pairs are now ("w", "or") and ("or", "d")
            - If the next most frequent pair is ("or", "d"), it merges again into ("w", "ord")
            - The final result is "w ord"
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        # e.g., ('w', 'o'), ('o', 'r'), ('r', 'd')
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            # Finds the most frequent (lowest rank) bigram in the current word
            # Assign a rank of infinity if a bigram isn't in self.bpe_ranks
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                # If first is followed by second, the pair is merged into a single token
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            # Convert new_word back to tuple
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        # Join with spaces between subwords
        word = ' '.join(word)
        # Add it to the cache for future lookups
        self.cache[token] = word
        return word

    def encode(self, text: str):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens: list):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(config_dir: str):
    with open(os.path.join(config_dir, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(config_dir, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
