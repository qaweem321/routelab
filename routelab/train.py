"""Fine-tune the router with LoRA.

python -m routelab.train                              # Qwen2.5-0.5B-Instruct on CPU, 1 epoch
python -m routelab.train --qlora --base Qwen/Qwen2.5-1.5B-Instruct   # 4-bit on a CUDA GPU
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from routelab.data import read
from routelab.labels import Label
from routelab.lora import DEFAULT_BASE, Collator, encode_example, load_tokenizer


class Encoded(Dataset):
    def __init__(self, rows: list[dict], tokenizer) -> None:  # noqa: ANN001
        self.items = [encode_example(tokenizer, r["text"], Label(r["route"], r["tool"])) for r in rows]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        return self.items[i]


def load_model(base: str, qlora: bool):  # noqa: ANN201
    if not qlora:
        return AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    if not torch.cuda.is_available():
        raise SystemExit("--qlora needs a CUDA GPU (bitsandbytes 4-bit). Use plain LoRA on a CPU.")
    from peft import prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto")
    return prepare_model_for_kbit_training(model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="models/router-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0, help="use only N training rows (quick runs)")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.base)
    tokenizer.padding_side = "right"
    train_rows = read(Path(args.data) / "train.jsonl")
    if args.limit:
        train_rows = train_rows[: args.limit]
    val_rows = read(Path(args.data) / "val.jsonl")

    model = load_model(args.base, args.qlora)
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    training = TrainingArguments(
        output_dir=f"{args.out}-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to=[],
        use_cpu=not torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=Encoded(train_rows, tokenizer),
        eval_dataset=Encoded(val_rows, tokenizer),
        data_collator=Collator(tokenizer.pad_token_id),
    )
    start = time.perf_counter()
    trainer.train()
    minutes = (time.perf_counter() - start) / 60

    out = Path(args.out)
    model.save_pretrained(out)
    tokenizer.padding_side = "left"
    tokenizer.save_pretrained(out)
    losses = [e for e in trainer.state.log_history if "loss" in e or "eval_loss" in e]
    info = {
        "base": args.base,
        "qlora": args.qlora,
        "train_rows": len(train_rows),
        "epochs": args.epochs,
        "lora_rank": args.rank,
        "learning_rate": args.lr,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "minutes": round(minutes, 1),
        "device": "cuda" if torch.cuda.is_available() else f"cpu ({torch.get_num_threads()} threads)",
        "log": losses,
    }
    (out / "training.json").write_text(json.dumps(info, indent=2))
    print(f"Saved the adapter to {out} after {minutes:.1f} min")


if __name__ == "__main__":
    main()
