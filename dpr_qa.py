#!/usr/bin/env python3
"""
Project 3: Dense Passage Retrieval (DPR) Open-Domain QA Pipeline
Builds a DPR-based QA system with FAISS vector index over Wikipedia passages.
Evaluated on Natural Questions (NQ) dataset end-to-end.
"""

import os
import json
import time
import numpy as np
import torch
import faiss
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizer,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    AutoTokenizer, AutoModelForQuestionAnswering,
    pipeline,
)
from datasets import load_dataset
from tqdm import tqdm
import string
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CTX_ENCODER_NAME = "facebook/dpr-ctx_encoder-single-nq-base"
Q_ENCODER_NAME   = "facebook/dpr-question_encoder-single-nq-base"
READER_NAME      = "deepset/roberta-base-squad2"   # extractive reader
PASSAGE_MAX_LEN  = 256
QUESTION_MAX_LEN = 64
TOP_K            = 5     # retrieved passages per question
EMBED_DIM        = 768   # DPR output dimension

# ──────────────────────────────────────────────
# 1. Load passage corpus
#    Using a small Wikipedia sample for demo;
#    for production use the full wiki_dpr dataset (~21M passages).
# ──────────────────────────────────────────────
print("\nLoading Wikipedia passage corpus (sample)...")
# Using psgs_w100 format: {id, title, text}
wiki_sample = load_dataset("wiki_dpr", "psgs_w100.nq.exact", split="train[:5000]",
                            trust_remote_code=True)
passages = [{"id": ex["id"], "title": ex["title"], "text": ex["text"]} for ex in wiki_sample]
print(f"Loaded {len(passages)} passages.")

# ──────────────────────────────────────────────
# 2. Context encoder — embed all passages
# ──────────────────────────────────────────────
print("\nLoading DPR context encoder...")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(CTX_ENCODER_NAME)
ctx_encoder   = DPRContextEncoder.from_pretrained(CTX_ENCODER_NAME).to(DEVICE)
ctx_encoder.eval()


def encode_passages(passages, batch_size=64):
    all_embeddings = []
    texts = [f"{p['title']} [SEP] {p['text']}" for p in passages]
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
        batch = texts[i: i + batch_size]
        enc   = ctx_tokenizer(
            batch,
            max_length=PASSAGE_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeddings = ctx_encoder(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            ).pooler_output  # (B, 768)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings).astype("float32")


print("Encoding passages...")
passage_embeddings = encode_passages(passages)
print(f"Passage embeddings shape: {passage_embeddings.shape}")

# ──────────────────────────────────────────────
# 3. Build FAISS index
# ──────────────────────────────────────────────
print("\nBuilding FAISS index...")
# Using flat inner-product index (exact search; use IVFFlat for large corpora)
faiss.normalize_L2(passage_embeddings)  # cosine similarity via L2-normalized IP
index = faiss.IndexFlatIP(EMBED_DIM)
index.add(passage_embeddings)
print(f"FAISS index built. Total vectors: {index.ntotal}")

# Optional: save / load index
FAISS_INDEX_PATH = "wiki_dpr_sample.index"
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"Index saved to {FAISS_INDEX_PATH}")


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

# ──────────────────────────────────────────────
# 4. Question encoder
# ──────────────────────────────────────────────
print("\nLoading DPR question encoder...")
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(Q_ENCODER_NAME)
q_encoder   = DPRQuestionEncoder.from_pretrained(Q_ENCODER_NAME).to(DEVICE)
q_encoder.eval()


def encode_question(question: str) -> np.ndarray:
    enc = q_tokenizer(
        question,
        max_length=QUESTION_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        embedding = q_encoder(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).pooler_output  # (1, 768)
    emb = embedding.cpu().numpy().astype("float32")
    faiss.normalize_L2(emb)
    return emb

# ──────────────────────────────────────────────
# 5. Retriever: retrieve top-k passages
# ──────────────────────────────────────────────
def retrieve(question: str, index: faiss.Index, passages: list, top_k: int = TOP_K):
    q_emb = encode_question(question)
    scores, indices = index.search(q_emb, top_k)  # (1, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(passages):
            results.append({
                "rank":   rank + 1,
                "score":  float(score),
                "id":     passages[idx]["id"],
                "title":  passages[idx]["title"],
                "text":   passages[idx]["text"],
            })
    return results


# ──────────────────────────────────────────────
# 6. Reader: extractive span prediction
# ──────────────────────────────────────────────
print("\nLoading extractive QA reader (RoBERTa-SQuAD2)...")
reader_pipeline = pipeline(
    "question-answering",
    model=READER_NAME,
    tokenizer=READER_NAME,
    device=0 if torch.cuda.is_available() else -1,
)


def read_answer(question: str, passages: list, reader, top_k: int = TOP_K):
    """Run extractive reader over retrieved passages; pick highest-confidence span."""
    candidates = []
    for p in passages[:top_k]:
        context = f"{p['title']}. {p['text']}"
        result  = reader(question=question, context=context, max_answer_len=50)
        candidates.append({
            "answer":  result["answer"],
            "score":   result["score"],
            "passage": p["title"],
        })
    best = max(candidates, key=lambda x: x["score"])
    return best, candidates


# ──────────────────────────────────────────────
# 7. Full pipeline: DPR retrieval + reader
# ──────────────────────────────────────────────
def qa_pipeline(question: str, index: faiss.Index, passages: list, reader):
    t0       = time.time()
    retrieved = retrieve(question, index, passages)
    answer, candidates = read_answer(question, retrieved, reader)
    elapsed  = time.time() - t0
    return {
        "question":   question,
        "answer":     answer["answer"],
        "confidence": answer["score"],
        "source":     answer["passage"],
        "latency_s":  round(elapsed, 3),
        "retrieved":  retrieved[:3],   # top-3 for display
        "candidates": candidates,
    }

# ──────────────────────────────────────────────
# 8. Evaluate on Natural Questions
# ──────────────────────────────────────────────
def normalize_text(s: str) -> str:
    """Lower-case, remove punctuation and extra spaces (NQ eval standard)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truths: list) -> bool:
    pred_norm = normalize_text(prediction)
    return any(pred_norm == normalize_text(gt) for gt in ground_truths)


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    p = len(common) / len(pred_tokens)
    r = len(common) / len(gold_tokens)
    return 2 * p * r / (p + r)


def evaluate_nq(questions: list, index: faiss.Index, passages: list, reader, max_samples: int = 100):
    """
    Evaluate DPR pipeline on NQ samples.
    Returns Exact Match (EM) and token-F1.
    """
    em_scores, f1_scores = [], []
    for sample in tqdm(questions[:max_samples], desc="Evaluating NQ"):
        q      = sample["question"]
        golds  = sample["answers"]   # list of acceptable answers
        result = qa_pipeline(q, index, passages, reader)
        pred   = result["answer"]
        em_scores.append(1 if exact_match(pred, golds) else 0)
        f1_scores.append(max(token_f1(pred, g) for g in golds))
    return {
        "exact_match": np.mean(em_scores),
        "token_f1":    np.mean(f1_scores),
        "n_samples":   len(em_scores),
    }


print("\nLoading Natural Questions validation split (sample)...")
nq_val = load_dataset("natural_questions", split="validation[:200]", trust_remote_code=True)

# NQ format: question.text, annotations.short_answers
nq_samples = []
for ex in nq_val:
    q_text  = ex["question"]["text"]
    answers = []
    for ann in ex["annotations"]["short_answers"]:
        answers.extend(ann["text"] if isinstance(ann["text"], list) else [ann["text"]])
    if answers:
        nq_samples.append({"question": q_text, "answers": answers})

print(f"NQ samples with short answers: {len(nq_samples)}")
nq_results = evaluate_nq(nq_samples, index, passages, reader_pipeline, max_samples=100)
print(f"\n--- NQ Evaluation Results (n={nq_results['n_samples']}) ---")
print(f"  Exact Match (EM) : {nq_results['exact_match']:.4f} ({nq_results['exact_match']*100:.2f}%)")
print(f"  Token F1         : {nq_results['token_f1']:.4f} ({nq_results['token_f1']*100:.2f}%)")

# ──────────────────────────────────────────────
# 9. Retrieval accuracy @ top-k
# ──────────────────────────────────────────────
def retrieval_accuracy_at_k(questions: list, index: faiss.Index, passages: list, k_vals=(1, 5, 20)):
    results = {k: [] for k in k_vals}
    for sample in tqdm(questions[:200], desc="Retrieval accuracy"):
        q      = sample["question"]
        golds  = [normalize_text(a) for a in sample["answers"]]
        retrieved = retrieve(q, index, passages, top_k=max(k_vals))
        for k in k_vals:
            hit = any(
                any(g in normalize_text(p["text"]) for g in golds)
                for p in retrieved[:k]
            )
            results[k].append(1 if hit else 0)
    print("\n--- Retrieval Accuracy ---")
    for k, hits in results.items():
        print(f"  Top-{k:2d} accuracy: {np.mean(hits):.4f} ({np.mean(hits)*100:.2f}%)")
    return results


retrieval_accuracy_at_k(nq_samples, index, passages)

# ──────────────────────────────────────────────
# 10. Interactive demo
# ──────────────────────────────────────────────
DEMO_QUESTIONS = [
    "Who wrote the novel Pride and Prejudice?",
    "What is the capital of France?",
    "When was the Eiffel Tower built?",
    "What programming language was Python named after?",
]

print("\n--- Interactive QA Demo ---")
for q in DEMO_QUESTIONS:
    result = qa_pipeline(q, index, passages, reader_pipeline)
    print(f"  Q: {result['question']}")
    print(f"  A: {result['answer']}  (confidence: {result['confidence']:.3f})")
    print(f"  Source: {result['source']}  | Latency: {result['latency_s']}s")
    print()