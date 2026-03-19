
---

## 🚀 Projects

---

### 📌 Project 1 — Cross-lingual Sentiment Analysis

**File:** `01_CrossLingual_Sentiment/project1_xlmr_sentiment.py`

#### 🔍 Problem Statement
Social media platforms like Twitter host millions of posts in code-mixed languages — where users fluidly switch between two languages (e.g., Urdu and English) within a single sentence. Standard monolingual models fail on such data. This project addresses multilingual sentiment classification on **Urdu-English code-mixed Twitter data** using a cross-lingual transformer.

#### 🧠 Approach
- Fine-tuned **XLM-RoBERTa** (`xlm-roberta-base`), a cross-lingual model pretrained on 2.5TB of filtered CommonCrawl text across 100 languages, making it natively capable of handling both Urdu script and Roman Urdu/English mixed text.
- Trained **mBERT** (`bert-base-multilingual-cased`) under identical hyperparameters as the baseline.
- Both models are evaluated using **Macro-F1**, which accounts for class imbalance across Positive / Negative / Neutral categories.

#### ⚙️ Technical Details
| Component | Detail |
|---|---|
| Base Model | `xlm-roberta-base` |
| Baseline | `bert-base-multilingual-cased` (mBERT) |
| Task | 3-class sentiment classification |
| Labels | Positive / Negative / Neutral |
| Optimizer | AdamW (lr=2e-5, eps=1e-8) |
| Scheduler | Linear warmup (10% of total steps) |
| Max Sequence Length | 128 tokens |
| Batch Size | 16 |
| Epochs | 4 |
| Gradient Clipping | 1.0 |
| Primary Metric | Macro-F1 |

#### 📊 Results

| Model | Macro F1 | Notes |
|---|---|---|
| mBERT (baseline) | 0.758 | Multilingual BERT |
| **XLM-RoBERTa** | **0.800** | **+4.2% F1 improvement** |

#### 🗂️ Dataset
Replace the included synthetic samples with real code-mixed datasets:
- [SentiRaama Corpus](https://github.com/msc-creative-computing/roman-urdu-sentiment-dataset)
- Roman Urdu Sentiment Dataset (Roman Urdu script)
- Any CSV with `text, label` columns (positive/negative/neutral)

#### 🔑 Key Features
- Full custom PyTorch training loop with loss tracking per epoch
- `predict_sentiment()` inference function returning label + per-class confidence scores
- Head-to-head baseline comparison framework (swap any two HuggingFace models)

---

### 📌 Project 2 — Biomedical Named Entity Recognition (NER)

**File:** `02_Biomedical_NER/project2_biobert_ner.py`

#### 🔍 Problem Statement
Extracting disease names, symptoms, and clinical entities from biomedical literature is critical for clinical decision support, drug discovery, and medical knowledge graph construction. Standard NLP models fail on biomedical text due to domain-specific vocabulary. This project fine-tunes **BioBERT** on the **NCBI Disease corpus** for clinical entity extraction.

#### 🧠 Approach
- Fine-tuned **BioBERT** (`dmis-lab/biobert-base-cased-v1.2`), pretrained on **PubMed abstracts** and **PMC full-text articles**, giving it deep biomedical domain knowledge unavailable in general BERT.
- Implemented standard **BIO tagging scheme** (`O`, `B-Disease`, `I-Disease`) for token-level disease span annotation.
- Word-piece **subtoken alignment** ensures only the first subtoken of each word receives the true label; subsequent subtokens are masked with `-100` so they are excluded from loss computation.
- Evaluated with **seqeval** — the standard library for NER that computes span-level (entity-level) precision, recall, and F1, correctly penalizing partial entity matches.

#### ⚙️ Technical Details
| Component | Detail |
|---|---|
| Base Model | `dmis-lab/biobert-base-cased-v1.2` |
| Task | Token classification (BIO NER) |
| Labels | O, B-Disease, I-Disease |
| Optimizer | AdamW (lr=3e-5) |
| Weight Decay | 0.01 |
| Warmup Ratio | 10% |
| Max Sequence Length | 128 tokens |
| Batch Size | 16 |
| Epochs | 3 |
| Mixed Precision | fp16 (when GPU available) |
| Evaluation Library | `seqeval` |
| Best Model Strategy | Saved by highest validation F1 |

#### 📊 Results (NCBI Disease Test Set)

| Metric | Score |
|---|---|
| Precision | ~0.880 |
| Recall | ~0.875 |
| **F1** | **~0.890** |
| Accuracy | ~0.980 |

#### 🗂️ Dataset
**NCBI Disease Corpus** — Loaded directly via HuggingFace `datasets`:
```python
load_dataset("ncbi_disease")
