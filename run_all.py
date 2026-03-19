#!/usr/bin/env python3
"""
run_all.py — Master runner for all three NLP projects.
Usage:
    python run_all.py --project 1   # XLM-RoBERTa Sentiment
    python run_all.py --project 2   # BioBERT NER
    python run_all.py --project 3   # DPR QA
    python run_all.py --project all # Run all sequentially
"""
import argparse
import subprocess
import sys

PROJECTS = {
    "1": ("project1_xlmr_sentiment.py", "XLM-RoBERTa Cross-lingual Sentiment"),
    "2": ("project2_biobert_ner.py",     "BioBERT Biomedical NER"),
    "3": ("project3_dpr_qa.py",          "DPR Open-Domain QA"),
}

def run(project_id: str):
    file, name = PROJECTS[project_id]
    print(f"\n{'='*70}")
    print(f"  Running Project {project_id}: {name}")
    print(f"{'='*70}\n")
    result = subprocess.run([sys.executable, file], check=True)
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    targets = list(PROJECTS.keys()) if args.project == "all" else [args.project]
    for pid in targets:
        run(pid)
    print("\nAll requested projects completed successfully.")