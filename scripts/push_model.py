"""
Push du modele fine-tune sur le Hugging Face Hub.

Deux modes :
  --mode adapter : pousse uniquement l'adapter LoRA (~140 Mo, rapide)
                   -> Utilisable avec le backend transformers (USE_VLLM=0).
  --mode merged  : merge le LoRA dans le modele de base et pousse le tout (~3,4 Go)
                   -> Utilisable avec vLLM (USE_VLLM=1) en production.

Usage:
    # Adapter seul (rapide, pour tests transformers)
    HF_TOKEN=hf_xxx python scripts/push_model.py \
        --mode adapter \
        --local-path "/mnt/c/Users/marin/Downloads/final (1)/final" \
        --repo Marintosti/chsa-triage-sft

    # Modele merge complet (pour vLLM)
    HF_TOKEN=hf_xxx python scripts/push_model.py \
        --mode merged \
        --local-path "/mnt/c/Users/marin/Downloads/final (1)/final" \
        --repo Marintosti/chsa-triage-merged \
        --base Qwen/Qwen3-1.7B-Base
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def push_adapter(local_path: Path, repo: str, token: str, private: bool) -> None:
    from huggingface_hub import HfApi, create_repo

    print(f"[adapter] Creation/verif du repo {repo} (private={private})")
    create_repo(repo_id=repo, token=token, private=private, exist_ok=True)

    print(f"[adapter] Upload de {local_path}")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo,
        commit_message="Upload LoRA adapter (SFT/DPO final)",
    )
    print(f"[adapter] OK -> https://huggingface.co/{repo}")


def push_merged(local_path: Path, repo: str, base: str, token: str, private: bool) -> None:
    """Charge le base + applique l'adapter LoRA + merge_and_unload + push."""
    import torch
    from huggingface_hub import create_repo
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merged] Chargement du modele de base : {base}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    print(f"[merged] Application de l'adapter LoRA : {local_path}")
    merged = PeftModel.from_pretrained(base_model, str(local_path))

    print("[merged] Merge des poids LoRA dans le modele de base ...")
    merged = merged.merge_and_unload()

    # Tokenizer : prend celui du dossier final (peut contenir un chat_template special)
    tokenizer_src = local_path if (local_path / "tokenizer.json").exists() else base
    print(f"[merged] Chargement du tokenizer : {tokenizer_src}")
    tok = AutoTokenizer.from_pretrained(str(tokenizer_src), trust_remote_code=True)

    print(f"[merged] Creation/verif du repo {repo} (private={private})")
    create_repo(repo_id=repo, token=token, private=private, exist_ok=True)

    print(f"[merged] Push du modele merge sur {repo} ...")
    merged.push_to_hub(repo, token=token, safe_serialization=True)
    tok.push_to_hub(repo, token=token)
    print(f"[merged] OK -> https://huggingface.co/{repo}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["adapter", "merged"], required=True)
    parser.add_argument("--local-path", required=True, help="Dossier contenant l'adapter LoRA")
    parser.add_argument("--repo", required=True, help="Repo HF cible (ex: user/chsa-triage-merged)")
    parser.add_argument("--base", default="Qwen/Qwen3-1.7B-Base", help="Modele de base (mode merged)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--public", action="store_true", help="Repo public (defaut: prive)")
    args = parser.parse_args()

    if not args.token:
        print("ERREUR: HF_TOKEN absent (--token ou variable d'env)", file=sys.stderr)
        return 1

    local_path = Path(args.local_path).expanduser().resolve()
    if not local_path.exists():
        print(f"ERREUR: dossier introuvable: {local_path}", file=sys.stderr)
        return 1
    if not (local_path / "adapter_config.json").exists():
        print(f"ERREUR: pas d'adapter_config.json dans {local_path}", file=sys.stderr)
        return 1

    private = not args.public

    if args.mode == "adapter":
        push_adapter(local_path, args.repo, args.token, private)
    else:
        push_merged(local_path, args.repo, args.base, args.token, private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
