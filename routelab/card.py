"""Write a Hugging Face model card for the adapter from its real training log and eval results.

python -m routelab.card                      # writes adapter/README.md
hf upload <you>/deskpilot-router-lora adapter   # optional: publish (needs an HF token)
"""

import argparse
import json
from pathlib import Path

CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
language: [en, ur]
pipeline_tag: text-generation
tags: [lora, peft, routing, function-calling, agents]
---

# deskpilot-router-lora

A LoRA adapter for **{base}** that routes a support-desk request to the right agent and tool for
[Deskpilot](https://github.com/qaweem321/deskpilot). It answers with one line of JSON:

```json
{{"route":"orders","tool":"refund_order"}}
```

Routes: `orders`, `tickets`, `knowledge`, `other`. Tools: `get_order`, `search_orders`, `cancel_order`,
`refund_order`, `get_ticket`, `search_tickets`, `create_ticket`, `update_ticket` (or `null`).

## Training

| | |
|---|---|
| Method | LoRA, rank {rank}, alpha {alpha}, on q/k/v/o projections; loss on the answer tokens only |
| Trainable parameters | {trainable:,} of {total:,} ({pct:.2f}%) |
| Data | {rows:,} synthetic requests from templates (English, some Urdu and Roman Urdu), balanced over 10 labels |
| Epochs / learning rate | {epochs} / {lr} (cosine) |
| Hardware | {device}, {minutes} minutes |

Train, validation and test sets use **different templates**, so the test measures phrasings the model never saw.

## Results

{results}

"deskpilot" is a small set of real requests from Deskpilot's own evaluation, written separately from the
training templates.

## Limits

- Trained on synthetic data for one store's support desk; other domains need new templates.
- It routes; it does not check permissions. Deskpilot enforces roles and confirmations after routing.
- Very short or ambiguous messages ("ok", "help") are guesses.

## Use

```python
from routelab.lora import LLMRouter
router = LLMRouter("{base}", "adapter")  # base model + this adapter
router.predict("Refund $20 on ORD-1002, the seam is torn")  # Label(route='orders', tool='refund_order')
```
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="adapter")
    p.add_argument("--results", default="reports/results.md")
    args = p.parse_args()
    adapter = Path(args.adapter)
    info = json.loads((adapter / "training.json").read_text())
    config = json.loads((adapter / "adapter_config.json").read_text())
    card = CARD.format(
        base=info["base"],
        rank=config["r"],
        alpha=config["lora_alpha"],
        trainable=info["trainable_parameters"],
        total=info["total_parameters"],
        pct=100 * info["trainable_parameters"] / info["total_parameters"],
        rows=info["train_rows"],
        epochs=info["epochs"],
        lr=info["learning_rate"],
        device=info["device"],
        minutes=info["minutes"],
        results=Path(args.results).read_text().strip(),
    )
    (adapter / "README.md").write_text(card)
    print(f"Wrote {adapter / 'README.md'}")


if __name__ == "__main__":
    main()
