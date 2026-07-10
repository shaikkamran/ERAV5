import os
import urllib.request
import urllib.parse
import json
import re

# Languages and their respective Wikipedia page titles for "India"
LANGS = {
    "en": {"title": "India", "url_lang": "en"},
    "hi": {"title": "भारत", "url_lang": "hi"},
    "te": {"title": "భారతదేశం", "url_lang": "te"},
    "ta": {"title": "இந்தியா", "url_lang": "ta"}
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def clean_text(text):
    # Remove references like [1], [12], [a], [citation needed], etc.
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[a-zA-Z]+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.I)
    
    # Remove weird characters but keep punctuation and spaces
    # (Especially keeping Indian script characters range: Devangari, Telugu, Tamil)
    # Standard clean up of double spaces, empty paragraphs, etc.
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = []
    for line in lines:
        if not line:
            continue
        # Remove markdown style sections if any (e.g. === History ===)
        if line.startswith('=') and line.endswith('='):
            continue
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)

def fetch_wikipedia_text(lang, title):
    encoded_title = urllib.parse.quote(title)
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles={encoded_title}&format=json&redirects=1"
    
    print(f"Fetching {title} from {lang}.wikipedia.org...")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
        
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise Exception(f"Failed to fetch content for {title} in {lang}")
        
    page_id = list(pages.keys())[0]
    if page_id == "-1":
        raise Exception(f"Page not found for {title} in {lang}")
        
    text = pages[page_id].get("extract", "")
    if not text:
        raise Exception(f"Empty text content returned for {title} in {lang}")
        
    return text

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for lang, info in LANGS.items():
        try:
            raw_text = fetch_wikipedia_text(info["url_lang"], info["title"])
            cleaned = clean_text(raw_text)
            
            # Print length and info
            words = re.findall(r'\w+', cleaned)
            print(f"Successfully fetched and cleaned {lang}. Characters: {len(cleaned)}, Words: {len(words)}")
            
            output_file = os.path.join(DATA_DIR, f"india_{lang}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Saved to {output_file}\n")
            
        except Exception as e:
            print(f"Error fetching {lang}: {e}\n")

if __name__ == "__main__":
    main()
