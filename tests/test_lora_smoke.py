"""Train a LoRA adapter for a few steps on a tiny random model: proves the whole pipeline runs
(encoding, label masking, training, saving, loading, generation) in seconds, without a real model."""

import subprocess
import sys

import pytest

from routelab.labels import Label
from routelab.lora import Collator, LLMRouter, encode_example, load_tokenizer

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(TINY)


def test_only_the_answer_is_trained(tokenizer):
    ex = encode_example(tokenizer, "Where is ORD-1004?", Label("orders", "get_order"))
    answer = [t for t in ex["labels"] if t != -100]
    assert tokenizer.decode(answer).startswith('{"route":"orders","tool":"get_order"}')
    assert ex["labels"][0] == -100 and len(ex["labels"]) == len(ex["input_ids"])


def test_collator_pads_with_ignored_labels(tokenizer):
    a = encode_example(tokenizer, "hi", Label("other", None))
    b = encode_example(tokenizer, "Where is ORD-1004 right now please?", Label("orders", "get_order"))
    batch = Collator(tokenizer.pad_token_id)([a, b])
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["attention_mask"][0].sum() == len(a["input_ids"])
    assert (batch["labels"][0][len(a["input_ids"]) :] == -100).all()


def test_train_save_load_generate(tmp_path):
    out = tmp_path / "adapter"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "routelab.train",
            "--base",
            TINY,
            "--out",
            str(out),
            "--limit",
            "32",
            "--epochs",
            "1",
            "--batch",
            "8",
        ],
        check=True,
    )
    assert (out / "adapter_config.json").exists() and (out / "training.json").exists()
    router = LLMRouter(TINY, out)
    results = router.predict_batch(["Where is ORD-1004?", "hi"], max_new_tokens=4)
    assert len(results) == 2  # a random tiny model outputs noise; the point is the plumbing works
