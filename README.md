# routelab

Fine-tuning a small open LLM with **LoRA** to route support-desk requests to the right agent and tool, and
measuring whether it was worth it. Built for [Deskpilot](https://github.com/qaweem321/deskpilot), whose
multi-agent assistant needs to know, for every message, which agent should handle it and which tool to call first:

```text
"Refund $20 on ORD-1002, the seam is torn"   ->  {"route":"orders","tool":"refund_order"}
"How long do refunds take?"                  ->  {"route":"knowledge","tool":null}
"Close TCK-2004"                             ->  {"route":"tickets","tool":"update_ticket"}
```

**Python, PyTorch, Hugging Face Transformers, PEFT (LoRA/QLoRA), scikit-learn, FastAPI.** Trained on a 2-core laptop CPU, no GPU.

![CI](https://github.com/qaweem321/routelab/actions/workflows/ci.yml/badge.svg)

## What's in it

| Step | File | What it does |
|---|---|---|
| Data | `routelab/templates.py`, `routelab/data.py` | 162 request templates over 10 labels (English, some Urdu and Roman Urdu), filled with ids, emails, amounts and reasons, plus realistic noise (casing, typos, prefixes). **Train, validation and test use different templates**, so the test measures phrasings the model never saw. |
| Baselines | `routelab/baselines.py` | Hand-written keyword rules, and TF-IDF (word + character n-grams) with logistic regression. |
| Fine-tuning | `routelab/train.py`, `routelab/lora.py` | LoRA on Qwen2.5-0.5B-Instruct with Hugging Face `Trainer` and PEFT. The loss covers only the answer tokens (prompt tokens are masked with -100). `--qlora` loads the base model in 4-bit NF4 with bitsandbytes on a CUDA GPU. |
| Evaluation | `routelab/evaluate.py`, `routelab/metrics.py` | Route accuracy, exact route + tool accuracy, macro-F1, share of valid outputs, latency, confusion matrix and the first errors per router. |
| Serving | `routelab/serve.py` | `POST /route` for Deskpilot's `ROUTER_MODE=remote`. |
| Model card | `routelab/card.py` | Writes a Hugging Face model card from the real training log and results. |
| Free GPU | `notebooks/qlora_free_gpu.ipynb` | The same scripts with QLoRA on Qwen2.5-1.5B, for a free Kaggle or Colab T4. |

## Results

Trained with LoRA (rank 16, 2.16M trainable parameters = 0.44% of the model) on 1,200 examples for one epoch:
**26.8 minutes on a 2-core laptop CPU**. Validation loss fell from 2.95 to 0.016. Full numbers, confusion matrices
and the first errors of every router are in [`reports/results.json`](reports/results.json).

**Held-out test set** (800 requests from templates the model never saw):

| Router | Route accuracy | Route + tool exact | Route macro-F1 | Valid outputs | ms / request (CPU) |
|---|---|---|---|---|---|
| Keyword rules | 73.9% | 68.0% | 0.651 | 100% | <1 |
| TF-IDF + logistic regression | 94.8% | 67.1% | 0.880 | 100% | 2 |
| Qwen2.5-0.5B, few-shot prompt (no training) | 63.0% | 52.0% | 0.533 | 77.2% | 1,600 |
| **Qwen2.5-0.5B + LoRA (this repo)** | **98.9%** | **95.6%** | **0.989** | **99.5%** | **396** |

**Deskpilot's real requests** (37 messages from [Deskpilot's evaluation suite](https://github.com/qaweem321/deskpilot/tree/main/backend/evals), written separately from the templates):

| Router | Route accuracy | Route + tool exact |
|---|---|---|
| Keyword rules* | 97.3% | 97.3% |
| TF-IDF + logistic regression | 94.6% | 94.6% |
| Qwen2.5-0.5B, few-shot prompt | 59.5% | 48.6% |
| **Qwen2.5-0.5B + LoRA** | **100%** | **100%** |

\* I wrote the keyword rules while looking at these requests, so that row is optimistic. The held-out test set is the fair comparison.

What the numbers say:
- **Fine-tuning beat prompting by a wide margin**: from 52% to 95.6% exact on unseen phrasings, and from 77% to 99.5% valid JSON.
  The few-shot prompt even shares some phrasings with the test set, which favours the base model.
- **It is also 4x faster than the prompted model.** The tuned model learned the task, so it needs a one-line prompt
  instead of the long instructions: 37 prompt tokens per request instead of 217. That is also what made CPU training practical.
- **Classic ML is a strong baseline for the route** (94.8%), but it can't tell the tool from unseen phrasings
  (67% exact). The model can, because it understands "reopen" or "compensate" without having seen those words in that template.
- **Remaining errors** (25 of 800): "Reopen TCK-…" is read as a lookup (`get_ticket`) instead of `update_ticket`,
  because no training template says "reopen". "Complaints from …" is sometimes taken as small talk. Both are fixed by adding templates.

### Does it help the real assistant?

Yes. Deskpilot's own 39-scenario evaluation, with the same local Qwen2.5-3B doing the agents' work, changing only the router:

| Deskpilot eval | Its LLM router | This fine-tuned router |
|---|---|---|
| Scenarios fully passed | 35 / 39 | **37 / 39** |
| Routing to the right agent | 97% | **100%** |
| Right tool chosen | 95% | **100%** |
| Safety scenarios | 13 / 13 | 13 / 13 |

The routing step also gets cheaper: a 0.5B model with a 37-token prompt, instead of the 3B chat model with a long instruction prompt.


## Run it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). PyTorch installs as the CPU build.

```bash
uv sync
```

```bash
uv run python -m routelab.data
```

```bash
uv run python -m routelab.train --limit 1200 --lr 3e-4
```

```bash
uv run python -m routelab.evaluate --routers keywords,tfidf,base,lora
```

```bash
uv run python -m routelab.card
```

Serve it for Deskpilot (then set `ROUTER_MODE=remote` and `ROUTER_URL=http://localhost:8700` in Deskpilot's `.env`):

```bash
uv run uvicorn --factory routelab.serve:create_app --port 8700
```

Tests (including a real 4-step LoRA training run on a tiny random model, so CI checks the whole pipeline in seconds):

```bash
uv run pytest
```

## Project structure

```
routelab/      data generation, baselines, LoRA training, evaluation, serving, model card
data/          generated train/val/test splits + Deskpilot's real requests (deskpilot_eval.jsonl)
adapter/       the trained LoRA adapter (about 9 MB) and its model card
reports/       evaluation results (markdown + JSON with errors and confusion matrices)
notebooks/     QLoRA on a free GPU
tests/         pytest
```

## License

MIT. The base model, Qwen2.5-0.5B-Instruct, is Apache-2.0.
