"""Scores for a router: route accuracy, route+tool accuracy, macro-F1, valid outputs, speed."""

import statistics
from collections import Counter
from dataclasses import asdict, dataclass

from routelab.labels import ROUTES, Label


@dataclass
class Report:
    name: str
    n: int
    route_accuracy: float
    exact_accuracy: float  # route and tool both right
    route_macro_f1: float
    valid_outputs: float
    ms_per_request: float
    confusion: dict[str, dict[str, int]]

    def row(self) -> dict:
        data = asdict(self)
        data.pop("confusion")
        return data


def macro_f1(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == p == label for g, p in zip(gold, pred, strict=True))
        fp = sum(p == label and g != label for g, p in zip(gold, pred, strict=True))
        fn = sum(g == label and p != label for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return statistics.fmean(scores)


def score(name: str, gold: list[Label], pred: list[Label | None], seconds: list[float]) -> Report:
    pred_routes = [p.route if p else "invalid" for p in pred]
    gold_routes = [g.route for g in gold]
    confusion: dict[str, dict[str, int]] = {r: dict(Counter()) for r in ROUTES}
    for g, p in zip(gold_routes, pred_routes, strict=True):
        confusion[g][p] = confusion[g].get(p, 0) + 1
    n = len(gold)
    return Report(
        name=name,
        n=n,
        route_accuracy=round(sum(g == p for g, p in zip(gold_routes, pred_routes, strict=True)) / n, 4),
        exact_accuracy=round(sum(g == p for g, p in zip(gold, pred, strict=True)) / n, 4),
        route_macro_f1=round(macro_f1(gold_routes, pred_routes, ROUTES), 4),
        valid_outputs=round(sum(p is not None for p in pred) / n, 4),
        ms_per_request=round(1000 * statistics.fmean(seconds), 1) if seconds else 0.0,
        confusion=confusion,
    )
