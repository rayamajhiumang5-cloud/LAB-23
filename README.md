# LAB-23
---
## P.R.I.M.E. README — Digital Portfolio Entry

# FedSpeak 2.0 — NLP Pipeline for Central Bank Communications

## Objective
Diagnose and correct a flawed NLP pipeline for FOMC minutes analysis,
then extend it with sentence-transformer embeddings and rate-decision prediction.

## Methodology
- **Diagnosed 3 planted errors**: naive `split()` tokenizer, Harvard GI
  (wrong financial dictionary), and TF-IDF with `min_df=1` / `max_df=1.0`.
- **Fixed preprocessing**: `re.sub` + `word_tokenize` → zero punctuation
  tokens, verified by assertion.
- **Switched to Loughran-McDonald (LM)** financial sentiment dictionary:
  reduced false-positive rate from ~50% to <10% on neutral financial terms.
- **Fixed TF-IDF**: `min_df=5`, `max_df=0.85`, bigrams, `sublinear_tf=True`
  — top terms no longer dominated by background words.
- **Sentence-transformer encoding**: `all-MiniLM-L6-v2` on truncated FOMC
  documents; compared cluster quality vs. TF-IDF + SVD.
- **TimeSeriesSplit evaluation** (5 folds): logistic regression for predicting
  Fed tightening cycles on both TF-IDF and embedding features.
- **Reusable module**: `src/fomc_sentiment.py` — `preprocess_fomc()`,
  `compute_lm_sentiment()`, `build_tfidf_matrix()`.

## Key Findings
- Sentence-transformer embeddings achieved higher mean AUC for rate-decision
  prediction, capturing contextual tone that keyword counts miss.
- LM dictionary reduced financial false-positive rate to <10% vs ~50% for GI.
- Bigram TF-IDF top features included meaningful Fed phrases ('interest rate',
  'price stability') absent with unigrams only.

## Stack
Python · scikit-learn · sentence-transformers · nltk · datasets · pandas
```
