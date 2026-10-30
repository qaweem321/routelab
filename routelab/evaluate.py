"""Compare routers on the held-out test set and on Deskpilot's real eval requests.

python -m routelab.evaluate --routers keywords,tfidf,base,lora
"""

import argparse
import json
import time
from pathlib import Path

from routelab.baselines import KeywordRouter, TfidfRouter
from routelab.data import read
from routelab.labels import Label
from routelab.lora import DEFAULT_BASE, LLMRouter, timed_batches
from routelab.metrics import Report, score


def run_simple(router, texts: list[str]) -> tuple[list[Label], list[float]]:  # noqa: ANN001
    preds, seconds = [], []
    for text in texts:
        start = time.perf_counter()
        preds.append(router.predict(text))
        seconds.append(time.perf_counter() - start)
    return preds, seconds


def table(reports: list[Report]) -> str:
    head = "| Router | Set | Route accuracy | Route + tool exact | Route macro-F1 | Valid outputs | ms / request |\n"
    head += "|---|---|---|---|---|---|---|\n"
    return head + "\n".join(
        f"| {r.name.split(' @ ')[0]} | {r.name.split(' @ ')[1]} | {r.route_accuracy:.1%} | {r.exact_accuracy:.1%} | "
        f"{r.route_macro_f1:.3f} | {r.valid_outputs:.1%} | {r.ms_per_request:.0f} |"
        for r in reports
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--routers", default="keywords,tfidf,base,lora")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--adapter", default="adapter")
    p.add_argument("--data", default="data")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="reports")
    args = p.parse_args()

    data = Path(args.data)
    sets = {"test": read(data / "test.jsonl"), "deskpilot": read(data / "deskpilot_eval.jsonl")}
    if args.limit:
        sets = {k: v[: args.limit] for k, v in sets.items()}
    train = read(data / "train.jsonl")

    reports: list[Report] = []
    errors: dict[str, list[dict]] = {}
    for name in args.routers.split(","):
        if name == "keywords":
            router, batched = KeywordRouter(), False
        elif name == "tfidf":
            router = TfidfRouter().fit([r["text"] for r in train], [Label(r["route"], r["tool"]) for r in train])
            batched = False
        elif name == "base":
            router, batched = LLMRouter(args.base), True
        elif name == "lora":
            router, batched = LLMRouter(args.base, args.adapter), True
        else:
            raise SystemExit(f"Unknown router {name}")

        for set_name, rows in sets.items():
            texts = [r["text"] for r in rows]
            gold = [Label(r["route"], r["tool"]) for r in rows]
            preds, seconds = timed_batches(router, texts) if batched else run_simple(router, texts)
            report = score(f"{name} @ {set_name}", gold, preds, seconds)
            reports.append(report)
            errors[report.name] = [
                {"text": t, "gold": g.key, "pred": p.key if p else None}
                for t, g, p in zip(texts, gold, preds, strict=True)
                if g != p
            ][:25]
            print(json.dumps(report.row()), flush=True)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"reports": [r.__dict__ for r in reports], "errors": errors}, indent=2, ensure_ascii=False)
    )
    (out / "results.md").write_text(table(reports) + "\n")
    print(table(reports))


if __name__ == "__main__":
    main()
