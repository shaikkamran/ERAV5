// Custom Multilingual BPE Tokenizer Browser Runtime

class BPETokenizer {
    constructor(vocab, merges, preTokenizePattern, graphemePattern) {
        this.vocab = vocab;
        this.merges = merges;
        
        // Fast ID lookup map
        this.charToId = {};
        vocab.forEach((token, idx) => {
            this.charToId[token] = idx;
        });

        this.preTokenizePattern = preTokenizePattern;
        this.graphemePattern = graphemePattern;
    }

    tokenize(text) {
        if (!text) {
            return { ids: [], tokens: [] };
        }
        
        // 1. Remove ZWNJ (\u200c) and ZWJ (\u200d) characters, then process spaces (replace with U+2581)
        const textCleaned = text.replace(/\u200c/g, '').replace(/\u200d/g, '');
        const textProcessed = textCleaned.replace(/ /g, ' ');

        // 2. Pre-tokenize into words, spaces, and isolated punctuation segments
        const segmentRegex = new RegExp(this.preTokenizePattern, 'g');
        const segments = textProcessed.match(segmentRegex) || [];

        const tokenizedIds = [];
        const tokenizedStrings = [];

        segments.forEach(segment => {
            // 3. Split segment into simple grapheme clusters (consonant + matras)
            const wordTokens = segment.match(new RegExp(this.graphemePattern, 'g')) || [];

            // 4. Apply BPE merges in the exact order of training
            this.merges.forEach(pair => {
                const parent = pair[0];
                const child = pair[1];
                const newWordTokens = [];
                let j = 0;
                
                while (j < wordTokens.length) {
                    if (j < wordTokens.length - 1 && wordTokens[j] === parent && wordTokens[j+1] === child) {
                        newWordTokens.push(parent + child);
                        j += 2;
                    } else {
                        newWordTokens.push(wordTokens[j]);
                        j += 1;
                    }
                }
                wordTokens.splice(0, wordTokens.length, ...newWordTokens);
            });

            // Convert merged tokens to vocabulary IDs
            wordTokens.forEach(t => {
                if (t in this.charToId) {
                    tokenizedIds.push(this.charToId[t]);
                    tokenizedStrings.push(t);
                } else {
                    // Skip or ignore unknown characters
                }
            });
        });

        return { ids: tokenizedIds, tokens: tokenizedStrings };
    }
}

// Clean word extractor for ratio denominator
function extractWords(text) {
    if (!text) return [];
    // Matches word cleaning regex in Python verify.py
    const cleaned = text.replace(/[\s.,!?;:\(\)\[\]\{\}"\'«»\-\–\—/\\\|*&^%$#@।॥_+=<>`~\u200b\u200c\u200d]+/g, ' ');
    return cleaned.trim().split(/\s+/).filter(w => w.length > 0);
}

// App Initialization
document.addEventListener("DOMContentLoaded", () => {
    // 1. Check data variables are present
    if (!window.TOKENIZER_DATA) {
        console.error("Error: window.TOKENIZER_DATA not loaded from tokenizer_data.js");
        return;
    }
    if (!window.SAMPLE_TEXTS) {
        console.error("Error: window.SAMPLE_TEXTS not loaded from sample_texts.js");
        return;
    }
    
    const data = window.TOKENIZER_DATA;
    const samples = window.SAMPLE_TEXTS;
    
    // Extract vocab and merges from either custom or standard HF format
    let vocab = data.vocab;
    let merges = data.merges;
    if (data.model) {
        const vocabDict = data.model.vocab;
        vocab = new Array(Object.keys(vocabDict).length);
        for (const [token, idx] of Object.entries(vocabDict)) {
            vocab[idx] = token;
        }
        merges = data.model.merges.map(m => Array.isArray(m) ? m : m.split(' '));
    }
    
    // Instantiate Tokenizer
    const tokenizer = new BPETokenizer(vocab, merges, data.pre_tokenize_pattern, data.grapheme_pattern);
    
    // 2. Populate Metrics Dashboard
    document.getElementById("score-val").innerText = data.score.toFixed(4);
    document.getElementById("base-vocab-val").innerText = vocab.length - merges.length;
    document.getElementById("merges-val").innerText = merges.length;
    
    // Populate ratios list
    const languages = data.langs;
    languages.forEach(lang => {
        const ratio = data.ratios[lang];
        const ratioTextEl = document.getElementById(`ratio-val-${lang}`);
        const ratioBarEl = document.getElementById(`ratio-bar-${lang}`);
        
        if (ratioTextEl) ratioTextEl.innerText = ratio.toFixed(6);
        
        // Animate progress bar (max range is around 4.0 for characters/word base)
        if (ratioBarEl) {
            setTimeout(() => {
                // Percentage based on max value of 4.0
                const percent = Math.min((ratio / 4.0) * 100, 100);
                ratioBarEl.style.width = `${percent}%`;
            }, 100);
        }
    });
    
    // Check if English complies with constraint
    const enStatusBadge = document.getElementById("en-status-badge");
    if (data.ratios.en <= 1.2) {
        enStatusBadge.className = "badge success";
    } else {
        enStatusBadge.className = "badge";
        enStatusBadge.style.borderColor = "#ff7b72";
        enStatusBadge.style.background = "rgba(255, 123, 114, 0.15)";
        enStatusBadge.style.color = "#ffa198";
        enStatusBadge.innerHTML = "English Ratio Check (FAIL &gt; 1.2)";
    }
    
    // 3. Downloader Button Event
    const downloadBtn = document.getElementById("download-tokenizer-btn");
    const jsonBlob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    downloadBtn.href = URL.createObjectURL(jsonBlob);
    downloadBtn.download = "tokenizer.json";
    
    // 4. Interactive Playground Logic
    const playgroundInput = document.getElementById("playground-input");
    const playgroundOutput = document.getElementById("playground-output");
    const playWords = document.getElementById("play-words");
    const playTokens = document.getElementById("play-tokens");
    const playRatio = document.getElementById("play-ratio");
    const loadSampleBtn = document.getElementById("load-sample-btn");
    const clearTextBtn = document.getElementById("clear-text-btn");
    const langTabs = document.getElementById("lang-tabs");
    
    let activeLang = "en";
    
    function runLiveTokenization() {
        const text = playgroundInput.value;
        const words = extractWords(text);
        const result = tokenizer.tokenize(text);
        
        // Update stats counters
        playWords.innerText = words.length;
        playTokens.innerText = result.tokens.length;
        
        const ratio = words.length > 0 ? (result.tokens.length / words.length) : 0.0;
        playRatio.innerText = ratio.toFixed(4);
        
        // Highlight English ratio limit violations in live stats
        if (activeLang === "en" && words.length > 0) {
            playRatio.style.color = ratio <= 1.2 ? "#56d364" : "#ff7b72";
        } else {
            playRatio.style.color = "var(--text-main)";
        }
        
        // Render color-coded tokens
        playgroundOutput.innerHTML = "";
        result.tokens.forEach((token, idx) => {
            const span = document.createElement("span");
            
            // Cycle through 5 different color classes
            const colorClass = `token-c${idx % 5}`;
            span.className = `token ${colorClass}`;
            
            // Replace U+2581 back to standard representation for readability
            let visibleText = token;
            if (token.startsWith(' ')) {
                // If it starts with space, draw a small dot representation
                const spaceIndicator = document.createElement("span");
                spaceIndicator.className = "token-space";
                spaceIndicator.innerText = "·";
                span.appendChild(spaceIndicator);
                visibleText = token.substring(1);
            }
            
            const textNode = document.createTextNode(visibleText);
            span.appendChild(textNode);
            playgroundOutput.appendChild(span);
        });
    }
    
    // Input listener
    playgroundInput.addEventListener("input", runLiveTokenization);
    
    // Clear button
    clearTextBtn.addEventListener("click", () => {
        playgroundInput.value = "";
        runLiveTokenization();
    });
    
    // Load sample excerpt button
    loadSampleBtn.addEventListener("click", () => {
        if (samples[activeLang]) {
            playgroundInput.value = samples[activeLang];
            runLiveTokenization();
        }
    });
    
    // Tab switching
    langTabs.addEventListener("click", (e) => {
        if (e.target.classList.contains("tab-btn")) {
            // Update active tab buttons
            document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
            e.target.classList.add("active");
            
            activeLang = e.target.getAttribute("data-lang");
            
            // Load and run new sample automatically
            playgroundInput.value = samples[activeLang] || "";
            runLiveTokenization();
        }
    });
    
    // Initial Run on start
    playgroundInput.value = samples["en"];
    runLiveTokenization();
});
