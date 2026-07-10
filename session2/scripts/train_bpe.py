import os
import re
import json
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Unicode ranges for Indic consonants and combining marks
INDIC_CONSONANTS = (
    r'[\u0904-\u0939\u0958-\u0961'  # Devanagari (Hindi)
    r'\u0c05-\u0c39\u0c58-\u0c61'  # Telugu
    r'\u0b85-\u0b9c\u0b9e-\u0ba9\u0baa-\u0bb9'  # Tamil
    r'\u0985-\u099c\u099e-\u0ba9\u0baa-\u0bb9]'  # Bengali
)

INDIC_COMBINING = (
    r'[\u0900-\u0903\u093e-\u094c\u094e-\u094f\u0951-\u0957\u0962-\u0963'  # Devanagari
    r'\u0c00-\u0c04\u0c3e-\u0c4c\u0c55-\u0c56\u0c62-\u0c63'  # Telugu
    r'\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcc\u0bd7]'  # Tamil
)

# Simple grapheme cluster regex pattern: groups consonant + matras (combining vowel marks)
# to prevent naive slicing, but keeps conjuncts split to minimize base vocab size.
SIMPLE_GRAPHEME_PATTERN = (
    rf'(?:{INDIC_CONSONANTS})(?:{INDIC_COMBINING})*'
    r'|.'
)

def split_graphemes(text):
    return re.findall(SIMPLE_GRAPHEME_PATTERN, text)

def extract_words(text):
    # Same word cleaning as the evaluator's clean words definition
    cleaned = re.sub(r'[\s.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~\u200b\u200c\u200d]+', ' ', text)
    return [w for w in cleaned.split() if w]

def pre_tokenize(text):
    # Replace space with U+2581 (lower one eighth block)
    text = text.replace(' ', ' ')
    
    # Punctuation characters to isolate
    punct = r'.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~'
    
    # Matches words, spaces, punctuation as isolated tokens
    pattern = rf' [^{punct} \n]+|[^{punct} \n]+| |[{punct}]|\n'
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
    # We will use "ta" (Tamil) as the default 4th language for the final widget
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
            sys.exit(1)
            
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        word_counts[lang] = len(extract_words(text))
        pre_tokens[lang] = pre_tokenize(text)
        print(f" - {lang:<5}: Characters = {len(text):<6} | Words (W) = {word_counts[lang]:<5} | Segments = {len(pre_tokens[lang])}")

    # Build base vocabulary of grapheme clusters
    all_graphemes = set()
    for lang in langs:
        for segment in pre_tokens[lang]:
            all_graphemes.update(split_graphemes(segment))
            
    sorted_graphemes = sorted(list(all_graphemes))
    print(f"\nBase Grapheme Vocab size: {len(sorted_graphemes)} tokens.")
    
    # Initialize vocabulary mappings
    char_to_id = {g: idx for idx, g in enumerate(sorted_graphemes)}
    id_to_token = {idx: g for idx, g in enumerate(sorted_graphemes)}
    
    # Initialize word frequencies with grapheme ID tuples
    word_freqs = {}
    for lang in langs:
        word_freqs[lang] = {}
        counts = {}
        for segment in pre_tokens[lang]:
            counts[segment] = counts.get(segment, 0) + 1
            
        for segment, freq in counts.items():
            graphemes = split_graphemes(segment)
            grapheme_ids = tuple(char_to_id[g] for g in graphemes)
            word_freqs[lang][grapheme_ids] = freq

    vocab_size = len(char_to_id)
    target_vocab_size = 10000
    merges = []
    
    print("\nStarting Grapheme-based BPE Training with Punctuation Isolation...")
    
    # Phase 1: English constraint <= 1.2
    print("\n[Phase 1] Compressing English to satisfy constraint (X1 <= 1.2)...")
    step = 0
    max_steps = target_vocab_size - vocab_size
    
    while vocab_size < target_vocab_size:
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
            print(" -> English fully compressed before reaching target ratio.")
            break
            
        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = vocab_size
        vocab_size += 1
        step += 1
        
        merges.append(best_pair)
        token_repr = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        id_to_token[new_id] = token_repr
        
        for l in langs:
            word_freqs[l] = {
                merge_tuple(word_tuple, best_pair, new_id): freq
                for word_tuple, freq in word_freqs[l].items()
            }

    # Phase 2: Balance the other three languages
    print("\n[Phase 2] Balancing Hindi, Telugu, and 4th language to optimize score...")
    other_langs = ["hi", "te", fourth_lang]
    
    while vocab_size < target_vocab_size:
        step += 1
        ratios = {}
        for l in other_langs:
            t_count = sum(len(word_tuple) * freq for word_tuple, freq in word_freqs[l].items())
            ratios[l] = t_count / word_counts[l]
            
        sorted_by_ratio = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
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
            
        new_id = vocab_size
        vocab_size += 1
        
        merges.append(best_pair)
        token_repr = id_to_token[best_pair[0]] + id_to_token[best_pair[1]]
        id_to_token[new_id] = token_repr
        
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

if __name__ == "__main__":
    main()
