import os
import re
import json
import sys

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        
        # Build a lookup table for merges to speed up tokenization
        # Maps (parent, child) -> parent+child
        self.merge_map = {}
        for parent, child in merges:
            self.merge_map[(parent, child)] = parent + child
            
    def tokenize(self, text):
        # Replace space with U+2581 (lower one eighth block)
        text_processed = text.replace(' ', ' ')
        
        # Split into words starting with U+2581, words without U+2581, and newlines
        pattern = r' [^ \n]+|[^ \n]+|\n'
        words = re.findall(pattern, text_processed)
        
        tokenized_ids = []
        for word in words:
            # Represent word as a list of its individual characters
            word_tokens = list(word)
            
            # Apply merges in the exact order they were trained
            for parent, child in self.merges:
                new_word_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and word_tokens[i] == parent and word_tokens[i+1] == child:
                        new_word_tokens.append(parent + child)
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens
                
            # Convert final merged string tokens to vocabulary IDs
            for t in word_tokens:
                if t in self.char_to_id:
                    tokenized_ids.append(self.char_to_id[t])
                else:
                    # Ignore unknown characters (or handle gracefully)
                    pass
        return tokenized_ids

def extract_words(text):
    # Same word cleaning as the training script (evaluator's clean words definition)
    cleaned = re.sub(r'[\s.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~\u200b\u200c\u200d]+', ' ', text)
    return [w for w in cleaned.split() if w]

def main():
    model_path = "/Users/kamran/Documents/ERAV5/session2/model/tokenizer.json"
    if not os.path.exists(model_path):
        print(f"Error: Tokenizer file not found at {model_path}")
        print("Please train the BPE tokenizer first by running 'python3 scripts/train_bpe.py'")
        sys.exit(1)
        
    with open(model_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        
    langs = tokenizer_data["langs"]
    vocab = tokenizer_data["vocab"]
    merges = tokenizer_data["merges"]
    
    print(f"Loaded tokenizer from {model_path}")
    print(f"Languages: {langs}")
    print(f"Vocab size: {len(vocab)} | Merges: {len(merges)}")
    
    tokenizer = BPETokenizer(vocab, merges)
    
    print("\nVerifying Tokenizer Ratios...")
    ratios = {}
    for lang in langs:
        filepath = f"/Users/kamran/Documents/ERAV5/session2/data/india_{lang}.txt"
        if not os.path.exists(filepath):
            print(f"Error: Data file for '{lang}' not found at {filepath}")
            sys.exit(1)
            
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        words = extract_words(text)
        word_count = len(words)
        
        # Run BPE tokenization
        token_ids = tokenizer.tokenize(text)
        token_count = len(token_ids)
        
        ratio = token_count / word_count
        ratios[lang] = ratio
        print(f" - {lang:<5}: Words (W) = {word_count:<5} | Tokens (T) = {token_count:<5} | Ratio (X) = {ratio:.6f}")
        
    # Calculate score
    x_vals = [ratios[l] for l in langs]
    x_min, x_max = min(x_vals), max(x_vals)
    score = 1000.0 / (x_max - x_min)
    
    print("\n" + "="*45)
    print(f"Verification Score: {score:.4f}")
    print(f"Formula: 1000 / ({x_max:.6f} - {x_min:.6f})")
    print(f"English Ratio (X1) <= 1.2 Constraint Check: {'PASS' if ratios['en'] <= 1.2 else 'FAIL'} ({ratios['en']:.6f})")
    print("="*45)

if __name__ == "__main__":
    main()
