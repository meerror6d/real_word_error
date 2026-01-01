#-----pseudocode--------

import pandas as pd
import math
import re
import unicodedata
from collections import Counter
from tqdm import tqdm

# === CONFIG ===
TRAIN_FILE = "train_tokens_simple_pos.xlsx"
TEST_FILE = "train_error_sentences.txt"    
OUTPUT_FILE = "train_realword_error_detection.xlsx"
THRESHOLD = 1e-4  # tweak for stricter or looser detection

# === UTILS ===
def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200B", "")
    text = re.sub(r'[^\S\n]+', ' ', text).strip()
    return text

def tokenize_code_mixed(sentence):
    return re.findall(r'[\u0980-\u09FF]+|[A-Za-z]+|[0-9]+|[^\s\w]', sentence)

def detect_lang(token):
    if re.fullmatch(r'[\u0980-\u09FF]+', token):
        return "BN"
    elif re.fullmatch(r'[A-Za-z]+', token):
        return "EN"
    elif re.fullmatch(r'[0-9]+', token):
        return "NUM"
    else:
        return "SYM"

# === BANGLA POS RULES ===
BN_PRONOUNS = ["আমি","তুমি","সে","আমরা","তারা","তিনি","তোমরা"]
BN_VERB_SUFFIXES = ["ছি","ছে","ল","লাম","বেন","বে"]
BN_NOUN_SUFFIXES = ["টা","রা","গুলো","গুলি","জন","বস্তু","জন্য"]
BN_ADJ_SUFFIXES = ["ওয়ালা","শীল","বান্ধব","যুক্ত","পূর্ণ"]
BN_ADV_SUFFIXES = ["ভাবে","ভাবেই","ধরে","ভাবের"]

def pos_bangla(token):
    if token in BN_PRONOUNS:
        return "PRON"
    elif token.endswith(tuple(BN_VERB_SUFFIXES)):
        return "VERB"
    elif token.endswith(tuple(BN_ADJ_SUFFIXES)):
        return "ADJ"
    elif token.endswith(tuple(BN_ADV_SUFFIXES)):
        return "ADV"
    elif token.endswith(tuple(BN_NOUN_SUFFIXES)):
        return "NOUN"
    else:
        return "NOUN"

def pos_english(token):
    token_lower = token.lower()
    if token_lower in ["i","you","he","she","we","they","it"]:
        return "PRON"
    elif token[0].isupper():
        return "PROPN"
    else:
        return "NOUN"

def simple_pos(token, lang):
    if lang == "BN":
        return pos_bangla(token)
    elif lang == "EN":
        return pos_english(token)
    elif lang == "NUM":
        return "NUM"
    else:
        return "PUNCT"

# === LOAD TRAIN MODEL ===
train_df = pd.read_excel(TRAIN_FILE)
sentences = (
    train_df.groupby("id")
      .apply(lambda x: [(str(w), str(p)) for w, p in zip(x["token"], x["pos"])])
      .tolist()
)

word_uni, word_bi, word_tri = Counter(), Counter(), Counter()
pos_uni, pos_bi, pos_tri = Counter(), Counter(), Counter()

for sent in sentences:
    words = ["<s>"] + [w for w,_ in sent] + ["</s>"]
    poss  = ["<s>"] + [p for _,p in sent] + ["</s>"]
    for i in range(len(words)):
        word_uni[words[i]] += 1
        pos_uni[poss[i]] += 1
        if i < len(words)-1:
            word_bi[(words[i], words[i+1])] += 1
            pos_bi[(poss[i], poss[i+1])] += 1
        if i < len(words)-2:
            word_tri[(words[i], words[i+1], words[i+2])] += 1
            pos_tri[(poss[i], poss[i+1], poss[i+2])] += 1

def trigram_probs(tri, bi, vocab):
    p = {}
    for (a,b,c),cnt in tri.items():
        denom = bi.get((a,b),0) + vocab
        p[(a,b,c)] = (cnt+1)/denom
    return p

Vw, Vp = len(word_uni), len(pos_uni)
word_tri_p = trigram_probs(word_tri, word_bi, Vw)
pos_tri_p  = trigram_probs(pos_tri, pos_bi, Vp)

def avg_log_prob(seq, tri_p, bi, V):
    tris = [(seq[i], seq[i+1], seq[i+2]) for i in range(len(seq)-2)]
    total = 0
    for t in tris:
        p = tri_p.get(t, 1/(bi.get((t[0],t[1]),0)+V))
        total += math.log(p)
    return total/len(tris) if tris else -999

# === LOAD TEST SENTENCES ===
with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_sentences = [normalize_text(line.strip()) for line in f if line.strip()]

# === DETECT ERRORS & TOKEN-LEVEL FLAGGING ===
results = []
for sid, sentence in enumerate(tqdm(test_sentences, desc="Checking sentences")):
    tokens = tokenize_code_mixed(sentence)
    tagged = [(t, simple_pos(t, detect_lang(t))) for t in tokens]

    words = ["<s>"] + [w for w,_ in tagged] + ["</s>"]
    poss  = ["<s>"] + [p for _,p in tagged] + ["</s>"]

    wprob = avg_log_prob(words, word_tri_p, word_bi, Vw)
    pprob = avg_log_prob(poss,  pos_tri_p,  pos_bi,  Vp)

    sentence_flag = "❌" if (pprob > -5 and wprob < math.log(THRESHOLD)) else "✅"

    # --- New: token-level error detection ---
    error_tokens = []
    for i in range(1, len(words)-1):
        trigram = (words[i-2], words[i-1], words[i])
        tri_prob = word_tri_p.get(trigram, 1/(word_bi.get((words[i-2], words[i-1]),0)+Vw))
        if tri_prob < THRESHOLD:
            error_tokens.append(words[i])

    results.append({
        "id": sid+1,
        "sentence": sentence,
        "tokens": " ".join(words[1:-1]),
        "pos_seq": " ".join(poss[1:-1]),
        "word_prob": round(math.exp(wprob),6),
        "pos_prob": round(math.exp(pprob),6),
        "sentence_flag": sentence_flag,
        "error_tokens": ", ".join(error_tokens)  # ✅ new column
    })

# === SAVE TO EXCEL ===
out = pd.DataFrame(results)
out.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")
print(f"✅ Detection complete. Saved to {OUTPUT_FILE}")
