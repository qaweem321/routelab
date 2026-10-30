"""LoRA fine-tuning of a small instruct model to output the route/tool JSON, plus inference.

Runs on a CPU (the default here) or a GPU. With --qlora the base model is loaded in 4-bit
(bitsandbytes, CUDA only), which is what you'd use on a free Kaggle/Colab T4 for bigger models.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from routelab.labels import Label, parse_label
from routelab.prompt import messages

DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"


def prompt_text(tokenizer: PreTrainedTokenizerBase, text: str, compact: bool = False) -> str:
    return tokenizer.apply_chat_template(messages(text, compact), tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer: PreTrainedTokenizerBase, text: str, label: Label, max_len: int = 128) -> dict:
    """Input = compact prompt + answer. Loss only on the answer tokens (prompt labels are -100)."""
    prompt_ids = tokenizer(prompt_text(tokenizer, text, compact=True), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(label.to_json() + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[: max_len - len(answer_ids)]
    return {
        "input_ids": prompt_ids + answer_ids,
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


@dataclass
class Collator:
    pad_id: int

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels, mask = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            mask.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batched generation
    return tokenizer


class LLMRouter:
    """Route with a causal LM (base model, base + LoRA adapter, or a merged model)."""

    def __init__(self, base: str, adapter: str | Path | None = None, threads: int | None = None) -> None:
        if threads:
            torch.set_num_threads(threads)
        self.tokenizer = load_tokenizer(
            str(adapter) if adapter and Path(adapter, "tokenizer_config.json").exists() else base
        )
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        self.model = model.eval()
        self.compact = adapter is not None  # fine-tuned: short prompt; base model: full instructions

    @torch.inference_mode()
    def predict_batch(self, texts: list[str], max_new_tokens: int = 24) -> list[tuple[Label | None, str]]:
        prompts = [prompt_text(self.tokenizer, t, self.compact) for t in texts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        texts_out = self.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return [(parse_label(t), t) for t in texts_out]

    def predict(self, text: str) -> Label | None:
        return self.predict_batch([text])[0][0]


def timed_batches(router: LLMRouter, texts: list[str], batch_size: int = 16) -> tuple[list[Label | None], list[float]]:
    preds: list[Label | None] = []
    seconds: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        start = time.perf_counter()
        results = router.predict_batch(chunk)
        per_item = (time.perf_counter() - start) / len(chunk)
        preds += [label for label, _ in results]
        seconds += [per_item] * len(chunk)
    return preds, seconds
