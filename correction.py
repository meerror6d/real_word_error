#-------pseudocode-------

import pandas as pd
import math
import re
from collections import Counter
from tqdm import tqdm

# =========================
# CONFIG
# =========================
TRAIN_FILE = "train_tokens_simple_pos.xlsx"
ERROR_FILE = "test_cleaned.txt"
BN_DICT_FILE = "bn_words.txt"
EN_DICT_FILE = "en_words.txt"
THRESHOLD = 1e-4
OUTPUT_FILE = "corrected_from_ngram.xlsx"

# =========================
# LOAD TRAIN DATA & BUILD N-GRAMS
# =========================
df = pd.read_excel(TRAIN_FILE)

# Convert to sentence-level token+POS lists
sentences = (
    df.groupby("id")
      .apply(lambda x: [(str(w), str(p)) for w, p in zip(x["token"], x["pos"]) if pd.notna(w) and pd.notna(p)])
      .tolist()
)

# Count word n-grams
word_uni, word_bi, word_tri = Counter(), Counter(), Counter()
for sent in sentences:
    words = ["<s>"] + [w for w,_ in sent] + ["</s>"]
    for i in range(len(words)):
        word_uni[words[i]] += 1
        if i < len(words)-1:
            word_bi[(words[i], words[i+1])] += 1
        if i < len(words)-2:
            word_tri[(words[i], words[i+1], words[i+2])] += 1

Vw = len(word_uni)

# Convert trigram counts to probabilities with add-1 smoothing
def trigram_probs(tri, bi, vocab_size):
    p = {}
    for (a,b,c), cnt in tri.items():
        denom = bi.get((a,b),0) + vocab_size
        p[(a,b,c)] = (cnt+1)/denom
    return p

word_tri_p = trigram_probs(word_tri, word_bi, Vw)

# Helper to compute avg log probability of trigrams
def avg_log_prob(seq, tri_p, bi, V):
    tris = [(seq[i],seq[i+1],seq[i+2]) for i in range(len(seq)-2)]
    total = 0
    for t in tris:
        p = tri_p.get(t, 1/(bi.get((t[0],t[1]),0)+V))
        total += math.log(p)
    return total/len(tris) if tris else -999

# =========================
# LOAD DICTIONARIES
# =========================
def load_wordlist(file):
    with open(file,'r',encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

bn_dict = load_wordlist(BN_DICT_FILE)
en_dict = load_wordlist(EN_DICT_FILE)

# Organize by word length for candidate selection
def dict_by_length(word_list):
    d = {}
    for w in word_list:
        l = len(w)
        d.setdefault(l, []).append(w)
    return d

bn_dict_len = dict_by_length(bn_dict)
en_dict_len = dict_by_length(en_dict)

# =========================
# UTILS
# =========================
def detect_lang(token):
    if re.fullmatch(r'[\u0980-\u09FF]+', token):
        return "BN"
    elif re.fullmatch(r'[A-Za-z]+', token):
        return "EN"
    elif re.fullmatch(r'[0-9]+', token):
        return "NUM"
    else:
        return "SYM"

def simple_pos(token, lang):
    # Simple rules: just for candidate filtering
    if lang=="BN": return "NOUN"
    elif lang=="EN": return "NOUN"
    elif lang=="NUM": return "NUM"
    else: return "PUNCT"

def tokenize_code_mixed(sentence):
    return re.findall(r'[\u0980-\u09FF]+|[A-Za-z]+|[0-9]+|[^\s\w]', sentence)

# =========================
# CORRECTION FUNCTION (NG-gram only - pseudocode)
# =========================
def correct_real_word_ngram(sentence, word_tri_p, word_bi, Vw, threshold=THRESHOLD, top_k=10):
    tokens = tokenize_code_mixed(sentence)
    corrected_tokens = tokens.copy()
    words_seq = ["<s>"] + tokens + ["</s>"]

    for i, word in enumerate(tokens):
        tri = (words_seq[i], words_seq[i+1], words_seq[i+2])
        prob = word_tri_p.get(tri, 1/(word_bi.get((tri[0],tri[1]),0)+Vw))

        if prob < threshold:
            lang = detect_lang(word)
            # Select candidate words of similar length ±1
            word_len = len(word)
            if lang=="BN":
                candidates = bn_dict_len.get(word_len, []) + bn_dict_len.get(word_len-1, []) + bn_dict_len.get(word_len+1, [])
            elif lang=="EN":
                candidates = en_dict_len.get(word_len, []) + en_dict_len.get(word_len-1, []) + en_dict_len.get(word_len+1, [])
            else:
                continue

            best_word = word
            best_prob = prob

            # Evaluate candidates
            for c in candidates:
                temp_seq = words_seq.copy()
                temp_seq[i+1] = c
                new_prob = avg_log_prob(temp_seq, word_tri_p, word_bi, Vw)
                if new_prob > best_prob:
                    best_prob = new_prob
                    best_word = c

            corrected_tokens[i] = best_word

    return " ".join(corrected_tokens)

# =========================
# PROCESS TEST FILE
# =========================
with open(ERROR_FILE,'r',encoding='utf-8') as f:
    test_sentences = [line.strip() for line in f if line.strip()]

rows = []
for sid, sent in enumerate(tqdm(test_sentences, desc="Correcting sentences")):
    corrected = correct_real_word_ngram(sent, word_tri_p, word_bi, Vw)
    rows.append({
        "id": sid+1,
        "original_sentence": sent,
        "corrected_sentence": corrected
    })

# =========================
# SAVE RESULTS
# =========================
out = pd.DataFrame(rows)
out.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"✅ Corrected sentences saved to {OUTPUT_FILE}")
