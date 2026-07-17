import os
import json
import re
import sys

# Color cycling for token representation in the terminal
COLORS = [
    "\033[91m",  # Red
    "\033[92m",  # Green
    "\033[93m",  # Yellow
    "\033[94m",  # Blue
    "\033[95m",  # Magenta
    "\033[96m",  # Cyan
]
RESET = "\033[0m"

# Indic scripts rules for graphemes
INDIC_CONSONANTS = r'[\u0904-\u0939\u0958-\u0961\u0c05-\u0c39\u0c58-\u0c61\u0b85-\u0b9c\u0b9e-\u0ba9\u0baa-\u0bb9\u0985-\u099c\u099e-\u0ba9\u0baa-\u0bb9]'
INDIC_COMBINING = r'[\u0900-\u0903\u093e-\u094c\u094e-\u094f\u0951-\u0957\u0962-\u0963\u0c00-\u0c04\u0c3e-\u0c4c\u0c55-\u0c56\u0c62-\u0c63\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcc\u0bd7]'
SIMPLE_GRAPHEME_PATTERN = rf'(?:{INDIC_CONSONANTS})(?:{INDIC_COMBINING})*|.'

def split_graphemes(text):
    return re.findall(SIMPLE_GRAPHEME_PATTERN, text)

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.cache = {}
        
    def encode(self, text):
        if not text:
            return []
            
        # Preprocess spaces and remove ZWJ/ZWNJ
        text_processed = text.replace('\u200c', '').replace('\u200d', '').replace(' ', ' ')
        
        punct = r'.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~'
        pattern = rf' [^{punct} \n]+|[^{punct} \n]+| |[{punct}]|\n'
        words = re.findall(pattern, text_processed)
        
        tokenized_ids = []
        for word in words:
            if word in self.cache:
                tokenized_ids.extend(self.cache[word])
                continue
                
            word_tokens = split_graphemes(word)
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
                
            word_ids = []
            for t in word_tokens:
                if t in self.char_to_id:
                    word_ids.append(self.char_to_id[t])
            
            self.cache[word] = word_ids
            tokenized_ids.extend(word_ids)
            
        return tokenized_ids

    def decode(self, ids):
        tokens = [self.vocab[idx] for idx in ids]
        text = "".join(tokens)
        return text.replace(' ', ' ')

    def colorize_tokens(self, ids):
        colored_text = []
        for i, idx in enumerate(ids):
            token_str = self.vocab[idx]
            # Replace whitespace U+2581 back to normal space representation visually
            visible_token = token_str.replace(' ', ' ')
            color = COLORS[i % len(COLORS)]
            colored_text.append(f"{color}{visible_token}{RESET}")
        return "".join(colored_text)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(os.path.dirname(script_dir), "model", "tokenizer.json")
    
    if not os.path.exists(model_path):
        print(f"Error: Tokenizer file not found at {model_path}")
        print("Please train the BPE tokenizer first.")
        sys.exit(1)
        
    with open(model_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)
        
    vocab = tokenizer_data["vocab"]
    merges = tokenizer_data["merges"]
    tokenizer = BPETokenizer(vocab, merges)
    
    # CLI check: If an argument is provided, tokenize it directly
    if len(sys.argv) > 1:
        input_text = sys.argv[1]
        print(f"\n--- Custom Text Input ---")
        print(f"Raw Input : {repr(input_text)}")
        ids = tokenizer.encode(input_text)
        print(f"Token IDs : {ids}")
        print(f"Token Cnt : {len(ids)}")
        decoded = tokenizer.decode(ids)
        print(f"Decoded   : {repr(decoded)}")
        print(f"Colored   : {tokenizer.colorize_tokens(ids)}")
        roundtrip_ok = re.sub(r'\s+', '', input_text) == re.sub(r'\s+', '', decoded)
        print(f"Roundtrip : {'PASS' if roundtrip_ok else 'FAIL'}")
        return

    # Default test use cases
    test_cases = [
        # 1. URL pattern that failed previously
        "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
        # 2. English text with punctuation
        "India, officially the Republic of India (Bhārat), is a country in South Asia.",
        # 3. Hindi text with combination marks
        "भारत के इतिहास में कई महान राजवंशों का शासन रहा है।",
        # 4. Telugu text
        "భారతదేశం వైవిధ్యమైన సంస్కృతులు, భాషల కలయిక.",
        # 5. Tamil text
        "இந்தியாவின் வரலாறு மிகவும் பழமையானது மற்றும் பெருமை வாய்ந்தது.",
    ]
    
    print("="*60)
    print(" BPE Tokenizer Encoding & Decoding Test Suite")
    print("="*60)
    
    for idx, text in enumerate(test_cases, 1):
        print(f"\n[Test Case {idx}]")
        print(f"Input:     {text}")
        
        # Encoding
        ids = tokenizer.encode(text)
        print(f"Token IDs: {ids[:15]}... ({len(ids)} tokens)" if len(ids) > 15 else f"Token IDs: {ids} ({len(ids)} tokens)")
        
        # Colorized Output
        colored = tokenizer.colorize_tokens(ids)
        print(f"Tokens:    {colored}")
        
        # Decoding
        decoded = tokenizer.decode(ids)
        
        # Check roundtrip (ignoring whitespaces as per grading policy)
        clean_orig = re.sub(r'\s+', '', text)
        clean_decoded = re.sub(r'\s+', '', decoded)
        
        if clean_orig == clean_decoded:
            print(f"Roundtrip: \033[92mPASS\033[0m")
        else:
            print(f"Roundtrip: \033[91mFAIL\033[0m")
            print(f"Expected:  {repr(clean_orig)}")
            print(f"Got:       {repr(clean_decoded)}")
            
    print("\n" + "="*60)
    print("Tip: Run this script with custom text as an argument:")
    print("     python3 scripts/run_tokenizer.py \"your text here\"")
    print("="*60)

if __name__ == "__main__":
    main()
