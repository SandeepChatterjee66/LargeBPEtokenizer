import re
import warnings
from collections import Counter
from typing import List, Dict, Tuple, Set
from datasets import load_dataset

class EnhancedBPETokenizer:
    def __init__(self, vocab_size: int = 10000, unknown_token: str = '<UNK>'):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], str] = {}
        self.base_vocab: Dict[str, int] = {}
        self.unknown_token = unknown_token
        self.space_marker = '◼️'  # Using a less likely character to avoid conflict Unicode block symbol
        
        # to Ensure unknown token is in base vocabulary
        self.base_vocab[self.unknown_token] = 0

    def get_stats(self, vocab: Counter) -> Counter:
        """ pair frequencies counter """
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[tuple(symbols[i:i+2])] += freq
        return pairs

    def merge_vocab(self, vocab: Counter, pair: Tuple[str, str]) -> Counter:
        """ vocabulary with merged tokens """
        new_vocab = Counter()
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            # More controlled replacement of pairs
            word_parts = word.split()
            new_word_parts = []
            i = 0
            
            while i < len(word_parts):
                if i < len(word_parts) - 1 and tuple(word_parts[i:i+2]) == pair:
                    new_word_parts.append(replacement)
                    i += 2
                else:
                    new_word_parts.append(word_parts[i])
                    i += 1
            
            new_word = ' '.join(new_word_parts)
            new_vocab[new_word] += vocab[word]
        
        return new_vocab

    def preprocess_text(self, text: str) -> List[str]:
        if not text or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace.")
        
        text = text.lower()
        words = re.findall(r'\b\w+\b|\S', text)
        return [f"{self.space_marker}{' '.join(list(word))}" for word in words]

    def train(self, texts: List[str]):
        """
        Train BPE tokenizer on given texts
        """
        if not texts:
            raise ValueError("No training texts provided.")
        
        vocab = Counter()
        for text in texts:
            try:
                tokens = self.preprocess_text(text)
                vocab.update(tokens)
            except ValueError as e:
                warnings.warn(f"Skipping invalid text: {e}")
        
        unique_chars = set(''.join(vocab.keys()))
        base_chars = [self.space_marker, self.unknown_token] + sorted(unique_chars)
        self.base_vocab = {char: idx for idx, char in enumerate(base_chars)}
        current_unique_tokens: Set[str] = set(vocab.keys())
        ordered_merges: List[Tuple[str, str]] = []
        
        while len(current_unique_tokens) < self.vocab_size:
            pairs = self.get_stats(vocab)
            # Stop
            if not pairs or len(self.vocab) >= self.vocab_size:
                break
            # frequent pair
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(vocab, best_pair)
            
            # Store merge rule-s in order
            self.merges[best_pair] = ''.join(best_pair)
            ordered_merges.append(best_pair)
            # unique tokens
            current_unique_tokens = set(' '.join(token.split()) for token in vocab.keys())
        
        # Create final vocabulary
        sorted_tokens = sorted(current_unique_tokens)
        self.vocab = {token: idx + len(self.base_vocab) for idx, token in enumerate(sorted_tokens)}

        # Ensure merge rules can be applied in order
        self.ordered_merges = ordered_merges

    def tokenize(self, text: str) -> List[int]:
        # Validate input
        if not text or not text.strip():
            warnings.warn("Empty text provided. Returning empty token list.")
            return []
        try:
            tokens = self.preprocess_text(text)
        except ValueError:
            warnings.warn("Invalid text preprocessing. Using unknown token.")
            return [self.base_vocab[self.unknown_token]]
        
        def get_token_ids(token: str) -> List[int]:
            current = token.split()
            for pair in self.ordered_merges:
                new_current = []
                i = 0
                while i < len(current):
                    if i < len(current) - 1 and tuple(current[i:i+2]) == pair:
                        new_current.append(self.merges[pair])
                        i += 2
                    else:
                        new_current.append(current[i])
                        i += 1
                current = new_current
            return [
                self.vocab.get(token, 
                    self.base_vocab.get(token, 
                        self.base_vocab[self.unknown_token]
                    )
                ) for token in current
            ]
        return [token_id for word in tokens for token_id in get_token_ids(word)]

    def decode(self, token_ids: List[int]) -> str:
        if not token_ids:
            warnings.warn("Empty token list provided. Returning empty string.")
            return ""
        reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        reverse_base_vocab = {idx: token for token, idx in self.base_vocab.items()}
        full_reverse_vocab = {**reverse_base_vocab, **reverse_vocab}
        tokens = [full_reverse_vocab.get(token_id, self.unknown_token) for token_id in token_ids]
        return ''.join(tokens).replace(self.space_marker, ' ').strip()

def train_tokenizer(texts: List[str], vocab_size: int = 5000) -> EnhancedBPETokenizer:
    tokenizer = EnhancedBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts)
    return tokenizer

def demonstrate_tokenizer(tokenizer: EnhancedBPETokenizer, sample_texts: List[str]):
    for sample_text in sample_texts:
        print("\nOriginal text:", sample_text)
        
        try:
            token_ids = tokenizer.tokenize(sample_text)
            print("Token IDs:", token_ids)
            decoded_text = tokenizer.decode(token_ids)
            print("Decoded text:", decoded_text)
        except Exception as e:
            print(f"Error processing text: {e}")

        
def load_wikipedia_texts(max_samples: int = None) -> List[str]:
    """ Load texts from the English Wikipedia dataset.
    """
    # Load the dataset
    dataset = load_dataset('lucadiliello/english_wikipedia', split='train')
    
    # Extract main texts
    texts = dataset['maintext']
    
    # Optional: limit samples for faster processing
    if max_samples is not None:
        texts = texts[:max_samples]
    
    return texts


# test
if __name__ == "__main__":
    sample_texts = [
        "Hello, this is a sample text to demonstrate BPE tokenization.",
        "Machine learning is fascinating and complex.",
        "Natural language processing helps computers understand human language.",
        "",  # Empty text to test error handling
        "12345 Special ch@r@cters and numb3rs"
    ]
    
    tokenizer = train_tokenizer(sample_texts, vocab_size=500)
    demonstrate_tokenizer(tokenizer, sample_texts)
    print("working...")
    
    
    # Load texts from Wikipedia
    wiki_texts = load_wikipedia_texts(max_samples=10000)  # Adjust max_samples as needed
    tokenizer = train_tokenizer(wiki_texts, vocab_size=5000)
    demonstrate_tokenizer(tokenizer, wiki_texts[:5])