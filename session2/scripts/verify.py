import os
import re
import json
import sys

class BPETokenizer:
    def __init__(self, vocab, merges, pre_tokenize_pattern=None, grapheme_pattern=None):
        self.vocab = vocab
        self.merges = merges
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.cache = {}
        
        # Load regex patterns dynamically from JSON or use defaults
        punct = r'.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~'
        self.pre_tokenize_pattern = pre_tokenize_pattern or rf' [^{punct} \n]+|[^{punct} \n]+| |[{punct}]|\n'
        
        indic_consonants = (
            r'[\u0904-\u0939\u0958-\u0961'
            r'\u0c05-\u0c39\u0c58-\u0c61'
            r'\u0b85-\u0b9c\u0b9e-\u0ba9\u0baa-\u0bb9'
            r'\u0985-\u099c\u099e-\u0ba9\u0baa-\u0bb9]'
        )
        indic_combining = (
            r'[\u0900-\u0903\u093e-\u094c\u094e-\u094f\u0951-\u0957\u0962-\u0963'
            r'\u0c00-\u0c04\u0c3e-\u0c4c\u0c55-\u0c56\u0c62-\u0c63'
            r'\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcc\u0bd7]'
        )
        self.grapheme_pattern = grapheme_pattern or rf'(?:{indic_consonants})(?:{indic_combining})*|.'
            
    def split_graphemes(self, text):
        return re.findall(self.grapheme_pattern, text)

    def tokenize(self, text):
        # Remove ZWNJ (\u200c) and ZWJ (\u200d) characters
        text = text.replace('\u200c', '').replace('\u200d', '')
        # Replace space with U+2581 (lower one eighth block)
        text_processed = text.replace(' ', ' ')

        # Split into words, spaces, and isolated punctuation segments
        words = re.findall(self.pre_tokenize_pattern, text_processed)
        
        tokenized_ids = []
        for word in words:
            if word in self.cache:
                tokenized_ids.extend(self.cache[word])
                continue
                
            # Represent word as a list of its simple grapheme clusters
            word_tokens = self.split_graphemes(word)
            
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
            word_ids = []
            for t in word_tokens:
                if t in self.char_to_id:
                    word_ids.append(self.char_to_id[t])
                else:
                    # Ignore unknown tokens
                    pass
            
            self.cache[word] = word_ids
            tokenized_ids.extend(word_ids)
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
    
    pre_tokenize_pattern = tokenizer_data.get("pre_tokenize_pattern")
    grapheme_pattern = tokenizer_data.get("grapheme_pattern")
    tokenizer = BPETokenizer(vocab, merges, pre_tokenize_pattern, grapheme_pattern)
    
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
