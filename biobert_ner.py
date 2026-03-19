#!/usr/bin/env python3
"""
Project 2: Biomedical Named Entity Recognition (NER)
Fine-tunes BioBERT on NCBI Disease corpus for clinical entity extraction.
Uses BIO tagging scheme. Evaluated with seqeval (precision / recall / F1).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import evaluate
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 3e-5

# BIO label scheme for NCBI disease corpus
LABEL_LIST  = ["O", "B-Disease", "I-Disease"]
LABEL2ID    = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL    = {i: l for l, i in LABEL2ID.items()}

print(f"Using device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

# ──────────────────────────────────────────────
# 1. Load NCBI Disease dataset from HuggingFace
# ──────────────────────────────────────────────
print("\nLoading NCBI Disease dataset...")
raw_dataset = load_dataset("ncbi_disease")
# raw_dataset has: train / validation / test splits
# Each example: {"id", "tokens": List[str], "ner_tags": List[int]}
# ner_tags: 0=O, 1=B-Disease, 2=I-Disease
print(raw_dataset)
print("\nSample example:")
print(raw_dataset["train"][0])

# ──────────────────────────────────────────────
# 2. Tokenize with word-piece alignment
# ──────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    all_labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        labels = []
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)           # special tokens → ignored
            elif word_id != prev_word_id:
                labels.append(label_ids[word_id])   # first subtoken → real label
            else:
                labels.append(-100)           # subsequent subtokens → ignored
            prev_word_id = word_id
        all_labels.append(labels)
    tokenized["labels"] = all_labels
    return tokenized


tokenized_dataset = raw_dataset.map(tokenize_and_align_labels, batched=True)
print("\nTokenized dataset:")
print(tokenized_dataset)

# ──────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
).to(DEVICE)

# ──────────────────────────────────────────────
# 4. Metrics (seqeval)
# ──────────────────────────────────────────────
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_preds = [
        [ID2LABEL[pred] for pred, lbl in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[lbl] for pred, lbl in zip(prediction, label) if lbl != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }

# ──────────────────────────────────────────────
# 5. Training arguments & Trainer
# ──────────────────────────────────────────────
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir              = "./biobert_ncbi_ner",
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    learning_rate           = LR,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    num_train_epochs        = EPOCHS,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    load_best_model_at_end  = True,
    metric_for_best_model   = "f1",
    logging_dir             = "./logs",
    logging_steps           = 50,
    fp16                    = torch.cuda.is_available(),
    seed                    = SEED,
    report_to               = "none",
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_dataset["train"],
    eval_dataset    = tokenized_dataset["validation"],
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# ──────────────────────────────────────────────
# 6. Train
# ──────────────────────────────────────────────
print("\nStarting BioBERT fine-tuning on NCBI Disease corpus...")
trainer.train()

# ──────────────────────────────────────────────
# 7. Evaluate on test set
# ──────────────────────────────────────────────
print("\n--- Test Set Evaluation ---")
test_results = trainer.evaluate(tokenized_dataset["test"])
print(f"  Precision : {test_results['eval_precision']:.4f}")
print(f"  Recall    : {test_results['eval_recall']:.4f}")
print(f"  F1        : {test_results['eval_f1']:.4f}")
print(f"  Accuracy  : {test_results['eval_accuracy']:.4f}")

# ──────────────────────────────────────────────
# 8. Detailed per-entity analysis
# ──────────────────────────────────────────────
def detailed_ner_eval(trainer, dataset, id2label):
    """Computes entity-level precision/recall/F1 with span-level matching."""
    predictions_output = trainer.predict(dataset)
    preds = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids
    true_preds = [
        [id2label[p] for p, l in zip(pred, lbl) if l != -100]
        for pred, lbl in zip(preds, labels)
    ]
    true_labels = [
        [id2label[l] for p, l in zip(pred, lbl) if l != -100]
        for pred, lbl in zip(preds, labels)
    ]
    seqeval_local = evaluate.load("seqeval")
    results = seqeval_local.compute(predictions=true_preds, references=true_labels)
    print("\nDetailed Entity Analysis:")
    if "Disease" in results:
        d = results["Disease"]
        print(f"  Disease → Precision: {d['precision']:.4f} | Recall: {d['recall']:.4f} | F1: {d['f1']:.4f} | Support: {d['number']}")
    print(f"  Overall  → Precision: {results['overall_precision']:.4f} | Recall: {results['overall_recall']:.4f} | F1: {results['overall_f1']:.4f}")
    return results

detailed_ner_eval(trainer, tokenized_dataset["test"], ID2LABEL)

# ──────────────────────────────────────────────
# 9. Inference
# ──────────────────────────────────────────────
def extract_diseases(text: str, model, tokenizer, id2label, device):
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
    tokens    = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    pred_ids  = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    pred_lbls = [id2label[p] for p in pred_ids]

    entities, current = [], []
    for token, label in zip(tokens, pred_lbls):
        if token in ("[CLS]", "[SEP]", "<s>", "</s>"):
            continue
        if label == "B-Disease":
            if current:
                entities.append(tokenizer.convert_tokens_to_string(current))
            current = [token]
        elif label == "I-Disease" and current:
            current.append(token)
        else:
            if current:
                entities.append(tokenizer.convert_tokens_to_string(current))
            current = []
    if current:
        entities.append(tokenizer.convert_tokens_to_string(current))
    return entities


demo_sentences = [
    "The patient presents with signs of Alzheimer disease and type 2 diabetes mellitus.",
    "Mutations in BRCA1 are associated with hereditary breast and ovarian cancer syndrome.",
    "The child was diagnosed with juvenile idiopathic arthritis at the age of 6.",
]

print("\n--- NER Inference Demo ---")
for sent in demo_sentences:
    diseases = extract_diseases(sent, model, tokenizer, ID2LABEL, DEVICE)
    print(f"  Text     : {sent}")
    print(f"  Diseases : {diseases}")
    print()