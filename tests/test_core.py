import random
from pathlib import Path

from fastapi.testclient import TestClient

from routelab.baselines import KeywordRouter, TfidfRouter
from routelab.data import build, read, split_templates
from routelab.labels import ALL_TOOLS, Label, parse_label
from routelab.metrics import macro_f1, score
from routelab.serve import create_app


def test_parse_label_accepts_only_the_label_space():
    assert parse_label('{"route":"orders","tool":"get_order"}') == Label("orders", "get_order")
    assert parse_label('Sure! {"route": "knowledge", "tool": null} done') == Label("knowledge", None)
    assert parse_label('{"route":"orders","tool":"create_ticket"}') is None  # tool of another agent
    assert parse_label('{"route":"orders","tool":null}') is None  # orders always needs a tool
    assert parse_label('{"route":"billing","tool":null}') is None
    assert parse_label("orders") is None


def test_templates_never_leak_between_splits():
    splits = split_templates(random.Random(13))
    seen = {name: {t for _, t in pairs} for name, pairs in splits.items()}
    assert not seen["train"] & seen["test"]
    assert not seen["train"] & seen["val"]
    assert not seen["val"] & seen["test"]


def test_dataset_is_balanced_and_valid():
    data = build(per_label={"train": 30, "val": 5, "test": 10})
    assert len(data["train"]) == 300 and len(data["test"]) == 100
    for row in data["train"]:
        label = Label(row["route"], row["tool"])
        assert parse_label(label.to_json()) == label
        assert row["tool"] is None or row["tool"] in ALL_TOOLS
    assert not {r["text"] for r in data["train"]} & {r["text"] for r in data["test"]}


def test_real_eval_set_is_labelled():
    rows = read(Path(__file__).parent.parent / "data" / "deskpilot_eval.jsonl")
    assert len(rows) >= 30
    assert all(parse_label(Label(r["route"], r["tool"]).to_json()) for r in rows)


def test_keyword_router():
    k = KeywordRouter()
    assert k.predict("Refund $20 on ORD-1002") == Label("orders", "refund_order")
    assert k.predict("How much of ORD-1006 can still be refunded?") == Label("orders", "get_order")
    assert k.predict("Close TCK-2004") == Label("tickets", "update_ticket")
    assert k.predict("Is shipping free?") == Label("knowledge", None)
    assert k.predict("hi") == Label("other", None)


def test_tfidf_learns_the_task():
    data = build(per_label={"train": 60, "val": 5, "test": 30})
    rows = data["train"]
    model = TfidfRouter().fit([r["text"] for r in rows], [Label(r["route"], r["tool"]) for r in rows])
    test = data["test"]
    correct = sum(model.predict(r["text"]) == Label(r["route"], r["tool"]) for r in test)
    assert correct / len(test) >= 0.65  # unseen templates; the full dataset does much better


def test_metrics():
    gold = [Label("orders", "get_order"), Label("knowledge", None), Label("other", None)]
    pred = [Label("orders", "refund_order"), Label("knowledge", None), None]
    report = score("x @ y", gold, pred, [0.1, 0.1, 0.1])
    assert report.route_accuracy == round(2 / 3, 4)
    assert report.exact_accuracy == round(1 / 3, 4)
    assert report.valid_outputs == round(2 / 3, 4)
    assert report.confusion["other"] == {"invalid": 1}
    assert macro_f1(["a", "b"], ["a", "b"], ("a", "b")) == 1.0


class FakeRouter:
    def predict(self, text):
        return None if text == "garbage" else Label("tickets", "get_ticket")


def test_serve_contract():
    client = TestClient(create_app(FakeRouter()))
    assert client.post("/route", json={"text": "Show TCK-2001"}).json() == {
        "route": "tickets",
        "tool": "get_ticket",
        "valid": True,
    }
    assert client.post("/route", json={"text": "garbage"}).json()["route"] == "knowledge"
    assert client.post("/route", json={"text": ""}).status_code == 422
