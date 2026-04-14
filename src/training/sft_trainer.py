"""
Semaine 2 - Fine-tuning supervise (SFT) avec LoRA.

Entraine le modele Qwen3-1.7B-Base sur le dataset medical SFT
en utilisant la technique LoRA pour une adaptation efficace.
"""

import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "configs/sft_config.yaml") -> dict:
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(config: dict):
    """Charge le modele et le tokenizer. LoRA sera applique par SFTTrainer via peft_config."""
    model_name = config["model"]["name"]
    dtype = getattr(torch, config["model"]["torch_dtype"])

    logger.info(f"Chargement du modele: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation=config["model"].get("attn_implementation"),
        device_map="auto",
        trust_remote_code=True,
    )

    # Configuration LoRA — sera passee au SFTTrainer (pas de get_peft_model ici)
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
    )

    return model, tokenizer, peft_config


def load_sft_data(config: dict):
    """Charge les datasets SFT."""
    train_path = str(PROJECT_ROOT / config["data"]["train_file"])
    eval_path = str(PROJECT_ROOT / config["data"]["eval_file"])

    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_path, split="train")

    logger.info(f"Train: {len(train_dataset)} exemples, Eval: {len(eval_dataset)} exemples")
    return train_dataset, eval_dataset


def train(config: dict) -> None:
    """Lance l'entrainement SFT."""
    model, tokenizer, peft_config = setup_model_and_tokenizer(config)
    train_dataset, eval_dataset = load_sft_data(config)

    training_cfg = config["training"]

    sft_config = SFTConfig(
        output_dir=str(PROJECT_ROOT / training_cfg["output_dir"]),
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        warmup_ratio=training_cfg["warmup_ratio"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        logging_steps=training_cfg["logging_steps"],
        eval_strategy=training_cfg["eval_strategy"],
        eval_steps=training_cfg["eval_steps"],
        save_strategy=training_cfg["save_strategy"],
        save_steps=training_cfg["save_steps"],
        save_total_limit=training_cfg["save_total_limit"],
        bf16=training_cfg["bf16"],
        max_seq_length=training_cfg["max_seq_length"],
        gradient_checkpointing=training_cfg["gradient_checkpointing"],
        report_to=training_cfg["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Demarrage de l'entrainement SFT...")
    trainer.train()

    # Sauvegarde du modele final
    final_path = str(PROJECT_ROOT / training_cfg["output_dir"] / "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modele SFT sauvegarde dans {final_path}")


if __name__ == "__main__":
    config = load_config()
    train(config)
