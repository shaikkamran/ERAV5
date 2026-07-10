import os
import re
import json
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def extract_words(text):
    # Replace punctuation and special symbols with spaces, keeping characters (including Indic Unicode combining marks)
    cleaned = re.sub(r'[\s.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~\u200b\u200c\u200d]+', ' ', text)
    return [w for w in cleaned.split() if w]

def pre_tokenize(text):
    # Replace space with U+2581 (lower one eighth block)
    text = text.replace(' ', ' ')
    # Split into words starting with U+2581, words without U+2581, and newlines
    pattern = r' [^ \n]+|[^ \n]+|\n'
    return re.findall(pattern, text)

def merge_tuple(word_tuple, pair, new_id):
    new_tuple = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i+1] == pair[1]:
            new_tuple.append(new_id)
            i += 2
        else:
            new_tuple.append(word_tuple[i])
            i += 1
    return tuple(new_tuple)

def main():
    # Allow user to specify the 4th language via command line
    # Default is "as" (Assamese) as it yields the highest score, but user can change it
    # fourth_lang = "as"
    fourth_lang = "ta"
    if len(sys.argv) > 1:
        fourth_lang = sys.argv[1]
        
    langs = ["en", "hi", "te", fourth_lang]
    
    # Load texts and compute word counts
    word_counts = {}
    pre_tokens = {}
    
    print(f"Loading corpus for languages: {langs}")
    for lang in langs:
        filepath = os.path.join(DATA_DIR, f"india_{lang}.txt")
        if not os.path.exists(filepath):
            print(f"Error: Corpus file for language '{lang}' not found at {filepath}")
            print("Please run scripts/fetch_data.py or ensure the file exists.")
            sys.exit(1)
            
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        word_counts[lang] = len(extract_words(text))
        pre_tokens[lang] = pre_tokenize(text)
        print(f" - {lang:<5}: Characters = {len(text):<6} | Words (W) = {word_counts[lang]:<5} | Pre-tokens = {len(pre_tokens[lang])}")

    # Build character vocabulary
    all_chars = set()
    for lang in langs:
        for token in pre_tokens[lang]:
            all_chars.update(token)
            
    sorted_chars = sorted(list(all_chars))
    print(f"\nBase alphabet size: {len(sorted_chars)} characters.")
    
    # Initialize vocabulary mappings
    char_to_id = {char: idx for idx, char in enumerate(sorted_chars)}
    id_to_token = {idx: char for idx, char in enumerate(sorted_chars)}
    
    # Initialize word frequencies with character ID tuples
    word_freqs = {}
    for lang in langs:
        word_freqs[lang] = {}
        counts = {}
        for token in pre_tokens[lang]:
            counts[token] = counts.get(token, 0) + 1
            
        for token, freq in counts.items():
            char_ids = tuple(char_to_id[c] for c in token)
            word_freqs[lang][char_ids] = freq

    vocab_size = len(char_to_id)
    target_vocab_size = 10000
    merges = []
    
    print("\nStarting BPE Training...")
    
    # Phase 1: Compress English until its tokenization ratio is <= 1.2
    # This is critical to satisfy the assignment requirement: X1 <= 1.2
    print("\n[Phase 1] Compressing English to satisfy constraint (X1 <= 1.2)...")
    step = 0
    max_steps = target_vocab_size - vocab_size
    
    while vocab_size < target_vocab_size:
        # Calculate current English ratio
        en_t_count = sum(len(word_tuple) * freq for word_tuple, freq in word_freqs["en"].items())
        en_ratio = en_t_count / word_counts["en"]
        
        if en_ratio <= 1.2000:
            print(f" -> Success: English ratio reached {en_ratio:.4f} <= 1.2000 at vocab size {vocab_size} (Step {step})")
            break
            
        # Count pairs in English
        pair_counts = {}
        for word_tuple, freq in word_freqs["en"].items():
            for pair in zip(word_tuple, word_tuple[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                
        if not pair_counts:
            print(" -> English fully compressed before reaching target ratio. Stopping.")
            break
            
        # Merge most frequent English pair
        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = vocab_size
        vocab_size += 1
        step += 1
        
        merges.append(best_pair)
        token_repr = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        id_to_token[new_id] = token_repr
        
        # Apply merge in all languages
        for l in langs:
            word_freqs[l] = {
                merge_tuple(word_tuple, best_pair, new_id): freq
                for word_tuple, freq in word_freqs[l].items()
            }

    # Phase 2: Balance the other three languages with the remaining budget
    # We want to minimize the maximum ratio of Hindi, Telugu, and the 4th language
    # to maximize the assignment score: 1000 / (X_max - X_min)
    print("\n[Phase 2] Balancing Hindi, Telugu, and 4th language to optimize score...")
    other_langs = ["hi", "te", fourth_lang]
    
    while vocab_size < target_vocab_size:
        step += 1
        
        # Calculate current ratios for the other languages
        ratios = {}
        for l in other_langs:
            t_count = sum(len(word_tuple) * freq for word_tuple, freq in word_freqs[l].items())
            ratios[l] = t_count / word_counts[l]
            
        # Find language with highest ratio
        sorted_by_ratio = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
        # Find most frequent pair in that language
        best_pair = None
        chosen_lang = None
        for l, ratio in sorted_by_ratio:
            pair_counts = {}
            for word_tuple, freq in word_freqs[l].items():
                for pair in zip(word_tuple, word_tuple[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
            if pair_counts:
                best_pair = max(pair_counts, key=pair_counts.get)
                chosen_lang = l
                break
                
        if best_pair is None:
            print(" -> No more pairs to merge. Stopping early.")
            break
            
        # Merge
        new_id = vocab_size
        vocab_size += 1
        
        merges.append(best_pair)
        token_repr = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        id_to_token[new_id] = token_repr
        
        # Apply merge to all languages
        for l in langs:
            word_freqs[l] = {
                merge_tuple(word_tuple, best_pair, new_id): freq
                for word_tuple, freq in word_freqs[l].items()
            }
            
        if step % 1000 == 0 or vocab_size == target_vocab_size:
            current_ratios = {}
            for l in langs:
                t_count = sum(len(word_tuple) * freq for word_tuple, freq in word_freqs[l].items())
                current_ratios[l] = t_count / word_counts[l]
            ratio_str = ", ".join([f"{l}: {current_ratios[l]:.4f}" for l in langs])
            print(f"Step {step}/{max_steps} | Vocab: {vocab_size} | Chosen Lang: {chosen_lang} | Pairs merged: {best_pair} -> {new_id} ({repr(token_repr)}) | Ratios: {ratio_str}")

    # Final stats
    print("\nTraining Complete!")
    final_ratios = {}
    for l in langs:
        t_count = sum(len(word_tuple) * freq for word_tuple, freq in word_freqs[l].items())
        final_ratios[l] = t_count / word_counts[l]
        print(f"Lang: {l:<5} | Words: {word_counts[l]:<5} | Tokens: {t_count:<5} | Ratio (X): {final_ratios[l]:.6f}")
        
    x_vals = [final_ratios[l] for l in langs]
    x_min, x_max = min(x_vals), max(x_vals)
    score = 1000.0 / (x_max - x_min)
    print(f"\nFinal Score: {score:.4f} (1000 / ({x_max:.6f} - {x_min:.6f}))")
    
    # Save model data
    saved_merges = [[id_to_token[p[0]], id_to_token[p[1]]] for p in merges]
    vocab_list = [id_to_token[i] for i in range(vocab_size)]
    
    tokenizer_data = {
        "langs": langs,
        "vocab": vocab_list,
        "merges": saved_merges,
        "ratios": final_ratios,
        "score": score
    }
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "model"), exist_ok=True)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "tokenizer.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nSaved tokenizer configuration to {model_path}")
    print(f"Saved vocabulary contains {len(vocab_list)} tokens.")
    print(f"Saved merges contains {len(saved_merges)} rules.")

if __name__ == "__main__":
    main()
