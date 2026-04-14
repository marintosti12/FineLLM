"""
Evaluation du modele medical.

Metriques :
- Perplexite sur le jeu de test
- Precision des reponses medicales
- Latence d'inference
- Comparaison base vs SFT vs DPO
"""

import json
import logging
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_model(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """Charge un modele et son tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def compute_perplexity(model, tokenizer, eval_data: list[str], max_length: int = 2048) -> float:
    """Calcule la perplexite moyenne sur les donnees d'evaluation."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in eval_data:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def measure_latency(model, tokenizer, prompts: list[str], max_new_tokens: int = 256) -> dict:
    """Mesure la latence d'inference."""
    model.eval()
    latencies = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            start = time.perf_counter()
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            end = time.perf_counter()
            latencies.append(end - start)

    return {
        "mean_latency_s": sum(latencies) / len(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "p50_latency_s": sorted(latencies)[len(latencies) // 2],
    }


def evaluate_model(model_path: str, eval_file: str) -> dict:
    """Evaluation complete d'un modele."""
    logger.info(f"Evaluation du modele: {model_path}")

    model, tokenizer = load_model(model_path)

    # Charger les donnees d'evaluation
    eval_dataset = load_dataset("json", data_files=eval_file, split="train")

    # Extraire les textes pour la perplexite
    eval_texts = []
    eval_prompts = []
    for item in eval_dataset:
        messages = item.get("messages", [])
        if messages:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            eval_texts.append(full_text)
            # Extraire le prompt (question utilisateur)
            user_msgs = [m["content"] for m in messages if m["role"] == "user"]
            if user_msgs:
                eval_prompts.append(user_msgs[0])

    # Perplexite
    logger.info("Calcul de la perplexite...")
    ppl = compute_perplexity(model, tokenizer, eval_texts[:100])

    # Latence
    logger.info("Mesure de la latence...")
    latency = measure_latency(model, tokenizer, eval_prompts[:20])

    results = {
        "model_path": model_path,
        "perplexity": ppl,
        "latency": latency,
        "num_eval_samples": len(eval_texts),
    }

    logger.info(f"Resultats: perplexite={ppl:.2f}, latence_moyenne={latency['mean_latency_s']:.2f}s")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default="data/sft/eval.jsonl")
    parser.add_argument("--output", type=str, default="outputs/eval_results.json")
    args = parser.parse_args()

    results = evaluate_model(args.model_path, str(PROJECT_ROOT / args.eval_file))

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Resultats sauvegardes dans {output_path}")
