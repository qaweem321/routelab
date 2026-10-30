"""Build the routing dataset: templates x slot values x light noise, split by template.

python -m routelab.data            # writes data/train.jsonl, data/val.jsonl, data/test.jsonl
"""

import json
import random
from collections import Counter
from pathlib import Path

from routelab.labels import Label
from routelab.templates import TEMPLATES

SLOTS = {
    "order": lambda r: f"ORD-{r.randint(1001, 1999)}",
    "ticket": lambda r: f"TCK-{r.randint(2001, 2999)}",
    "email": lambda r: (
        f"{r.choice(['sara', 'omar', 'lena', 'mei', 'tom', 'priya', 'jonas', 'aisha', 'carlos', 'dan'])}@{r.choice(['example.com', 'mail.test', 'shop.test'])}"
    ),
    "amount": lambda r: (
        r.choice(["$", "$", "USD ", ""]) + str(r.choice([5, 10, 12.5, 15, 20, 25, 30, 45, 49.99, 60, 80, 120]))
    ),
    "reason": lambda r: r.choice(
        [
            "the seam is torn",
            "it arrived damaged",
            "the customer changed their mind",
            "wrong size",
            "duplicate order",
            "a strap was missing",
            "it arrived late",
            "ordered by mistake",
        ]
    ),
    "status": lambda r: r.choice(["pending", "shipped", "delivered", "cancelled", "refunded"]),
    "tstatus": lambda r: r.choice(["open", "pending", "closed"]),
    "priority": lambda r: r.choice(["low", "normal", "high", "urgent"]),
    "issue": lambda r: r.choice(
        [
            "parcel never arrived",
            "tracking stuck for 5 days",
            "jacket zip broken",
            "wants an invoice",
            "charged twice",
            "wrong item sent",
        ]
    ),
    "product": lambda r: r.choice(["backpacks", "trekking poles", "jackets", "stoves"]),
    "payment": lambda r: r.choice(["PayPal", "Apple Pay", "Amex", "bank transfer"]),
}
PREFIXES = ["", "", "", "hey, ", "pls ", "Quick one: ", "Can you help? ", "Hi, "]
SPLITS = {"train": 0.7, "val": 0.1, "test": 0.2}


def fill(template: str, rng: random.Random) -> str:
    text = template
    for name, make in SLOTS.items():
        while "{" + name + "}" in text:
            text = text.replace("{" + name + "}", make(rng), 1)
    return text


def noisy(text: str, rng: random.Random) -> str:
    """Light, realistic noise: casing, a prefix, a dropped question mark, one typo."""
    text = rng.choice(PREFIXES) + text
    roll = rng.random()
    if roll < 0.15:
        text = text.lower()
    elif roll < 0.25:
        text = text.rstrip("?!.")
    if rng.random() < 0.1 and len(text) > 8:
        i = rng.randrange(1, len(text) - 1)
        if text[i].isalpha() and text[i + 1].isalpha():
            text = text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    return text


def split_templates(rng: random.Random) -> dict[str, list[tuple[str, str]]]:
    """Assign whole templates to splits (per label), so no phrasing leaks into the test set."""
    out: dict[str, list[tuple[str, str]]] = {s: [] for s in SPLITS}
    for key, templates in TEMPLATES.items():
        shuffled = templates[:]
        rng.shuffle(shuffled)
        n_test = max(2, round(len(shuffled) * SPLITS["test"]))
        n_val = max(1, round(len(shuffled) * SPLITS["val"]))
        out["test"] += [(key, t) for t in shuffled[:n_test]]
        out["val"] += [(key, t) for t in shuffled[n_test : n_test + n_val]]
        out["train"] += [(key, t) for t in shuffled[n_test + n_val :]]
    return out


def label_of(key: str) -> Label:
    route, tool = key.split("/")
    return Label(route, None if tool == "-" else tool)


def build(seed: int = 13, per_label: dict[str, int] | None = None) -> dict[str, list[dict]]:
    per_label = per_label or {"train": 300, "val": 40, "test": 80}
    rng = random.Random(seed)
    datasets: dict[str, list[dict]] = {}
    for split, pairs in split_templates(rng).items():
        rows, seen = [], set()
        by_key: dict[str, list[str]] = {}
        for key, template in pairs:
            by_key.setdefault(key, []).append(template)
        for key, templates in sorted(by_key.items()):
            label = label_of(key)
            attempts = 0
            count = 0
            while count < per_label[split] and attempts < per_label[split] * 20:
                attempts += 1
                text = noisy(fill(rng.choice(templates), rng), rng)
                if text in seen:
                    continue
                seen.add(text)
                rows.append({"text": text, "route": label.route, "tool": label.tool})
                count += 1
        rng.shuffle(rows)
        datasets[split] = rows
    return datasets


def write(datasets: dict[str, list[dict]], folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for split, rows in datasets.items():
        with (folder / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    data = build()
    write(data, Path("data"))
    for split, rows in data.items():
        counts = Counter(f"{r['route']}/{r['tool'] or '-'}" for r in rows)
        print(split, len(rows), dict(sorted(counts.items())))
