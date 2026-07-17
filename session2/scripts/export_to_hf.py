import json
import os
import sys

def main():
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.normalizers import Sequence, Replace
        from tokenizers.pre_tokenizers import Split
        from tokenizers.decoders import Replace as ReplaceDecoder
    except ImportError:
        print("Error: tokenizers library not found. Run pip install tokenizers first.")
        sys.exit(1)

    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
    model_path = os.path.join(model_dir, "tokenizer.json")

    if not os.path.exists(model_path):
        print(f"Error: Custom tokenizer config not found at {model_path}")
        sys.exit(1)

    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Parse merges and vocab
    vocab = data["vocab"]
    vocab_dict = {t: i for i, t in enumerate(vocab)}
    merges_list = [tuple(m) for m in data["merges"]]

    # 2. Create the standard BPE model
    model = BPE(vocab=vocab_dict, merges=merges_list)

    # 3. Instantiate tokenizer
    tokenizer = Tokenizer(model)

    # 4. Set up Normalizer (strips ZWJ/ZWNJ, replaces spaces with U+2581)
    normalizer = Sequence([
        Replace("\u200c", ""),
        Replace("\u200d", ""),
        Replace(" ", " ")
    ])
    tokenizer.normalizer = normalizer

    # 5. Set up Pre-Tokenizer (isolates words, punctuation, and splits characters)
    import re
    from tokenizers import Regex
    from tokenizers.pre_tokenizers import Sequence as PreSequence
    
    def decode_escapes(s):
        return re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), s)

    pre_tokenize_pattern = decode_escapes(data.get("pre_tokenize_pattern"))
    grapheme_pattern = decode_escapes(data.get("grapheme_pattern"))
    
    pre_tokenizer = PreSequence([
        Split(Regex(pre_tokenize_pattern), behavior="isolated"),
        Split(Regex(grapheme_pattern), behavior="isolated")
    ])
    tokenizer.pre_tokenizer = pre_tokenizer

    # 6. Set up Decoder (converts U+2581 back to normal spaces)
    decoder = ReplaceDecoder(" ", " ")
    tokenizer.decoder = decoder

    # 7. Save standard tokenizer back to the model path
    tokenizer.save(model_path)

    # 8. Load standard tokenizer JSON and inject custom metadata back
    with open(model_path, "r", encoding="utf-8") as f:
        hf_data = json.load(f)
        
    hf_data["langs"] = data.get("langs")
    hf_data["ratios"] = data.get("ratios")
    hf_data["score"] = data.get("score")
    hf_data["pre_tokenize_pattern"] = pre_tokenize_pattern
    hf_data["grapheme_pattern"] = grapheme_pattern
    
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(hf_data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully exported tokenizer to standard Hugging Face format with metadata at {model_path}!")

if __name__ == "__main__":
    main()
