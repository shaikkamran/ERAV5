import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LANGS = ["en", "hi", "te", "ta"]

def extract_words(text):
    # Replace punctuation and special symbols with spaces, keeping characters (including Indic Unicode combining marks)
    cleaned = re.sub(r'[\s.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~\u200b\u200c\u200d]+', ' ', text)
    return [w for w in cleaned.split() if w]

def analyze_file(filename):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    char_count = len(text)
    unique_chars = sorted(list(set(text)))
    unique_char_count = len(unique_chars)
    
    words = extract_words(text)
    word_count = len(words)
    unique_word_count = len(set(words))
    
    return {
        "char_count": char_count,
        "unique_char_count": unique_char_count,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
    }

def main():
    print(f"{'Lang':<6} | {'Chars':<10} | {'Unique Chars':<15} | {'Words':<12} | {'Unique Words':<15}")
    print("-" * 70)
    for lang in LANGS:
        filename = f"india_{lang}.txt"
        stats = analyze_file(filename)
        print(f"{lang:<6} | {stats['char_count']:<10} | {stats['unique_char_count']:<15} | {stats['word_count']:<12} | {stats['unique_word_count']:<15}")

if __name__ == "__main__":
    main()
