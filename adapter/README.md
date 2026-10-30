---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
language: [en, ur]
pipeline_tag: text-generation
tags: [lora, peft, routing, function-calling, agents]
---

# deskpilot-router-lora

A LoRA adapter for **Qwen/Qwen2.5-0.5B-Instruct** that routes a support-desk request to the right agent and tool for
[Deskpilot](https://github.com/qaweem321/deskpilot). It answers with one line of JSON:

```json
{"route":"orders","tool":"refund_order"}
```

Routes: `orders`, `tickets`, `knowledge`, `other`. Tools: `get_order`, `search_orders`, `cancel_order`,
`refund_order`, `get_ticket`, `search_tickets`, `create_ticket`, `update_ticket` (or `null`).

## Training

| | |
|---|---|
| Method | LoRA, rank 16, alpha 32, on q/k/v/o projections; loss on the answer tokens only |
| Trainable parameters | 2,162,688 of 496,195,456 (0.44%) |
| Data | 1,200 synthetic requests from templates (English, some Urdu and Roman Urdu), balanced over 10 labels |
| Epochs / learning rate | 1.0 / 0.0003 (cosine) |
| Hardware | cpu (2 threads), 26.8 minutes |

Train, validation and test sets use **different templates**, so the test measures phrasings the model never saw.

## Results

| Router | Set | Route accuracy | Route + tool exact | Route macro-F1 | Valid outputs | ms / request |
|---|---|---|---|---|---|---|
| keywords | test | 73.9% | 68.0% | 0.651 | 100.0% | 0 |
| keywords | deskpilot | 97.3% | 97.3% | 0.905 | 100.0% | 0 |
| tfidf | test | 94.8% | 67.1% | 0.880 | 100.0% | 2 |
| tfidf | deskpilot | 94.6% | 94.6% | 0.880 | 100.0% | 2 |
| lora | test | 98.9% | 95.6% | 0.989 | 99.5% | 396 |
| lora | deskpilot | 100.0% | 100.0% | 1.000 | 100.0% | 457 |
| base | test | 63.0% | 52.0% | 0.533 | 77.2% | 1600 |
| base | deskpilot | 59.5% | 48.6% | 0.374 | 78.4% | 1752 |

"deskpilot" is a small set of real requests from Deskpilot's own evaluation, written separately from the
training templates.

## Limits

- Trained on synthetic data for one store's support desk; other domains need new templates.
- It routes; it does not check permissions. Deskpilot enforces roles and confirmations after routing.
- Very short or ambiguous messages ("ok", "help") are guesses.

## Use

```python
from routelab.lora import LLMRouter

router = LLMRouter("Qwen/Qwen2.5-0.5B-Instruct", "adapter")  # base model + this adapter
router.predict("Refund $20 on ORD-1002, the seam is torn")  # Label(route='orders', tool='refund_order')
```
