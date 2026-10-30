"""Two baselines a fine-tuned model has to beat: hand-written keyword rules, and TF-IDF features
with logistic regression (classic ML, trained in seconds)."""

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from routelab.labels import Label

_ORDER = re.compile(r"\bORD-\d+", re.I)
_TICKET = re.compile(r"\bTCK-\d+", re.I)
_EMAIL = re.compile(r"\S+@\S+")


class KeywordRouter:
    """Rules in the spirit of Deskpilot's keyword fallback, extended to pick a tool."""

    def predict(self, text: str) -> Label:
        t = text.lower()
        if re.match(r"^\s*(hi|hello|hey|thanks|thank you|salam|shukriya|ok|good morning)\b", t) and len(t) < 40:
            return Label("other", None)
        if _TICKET.search(text) or re.search(r"\b(ticket|tickets|case|complaint)s?\b", t):
            if re.search(r"\b(open a|create|new|raise|log|file|make|start|record)\b", t) and _EMAIL.search(text):
                return Label("tickets", "create_ticket")
            if re.search(r"\b(close|reopen|mark|set|escalate|priority|lower|bump|update|band)\b", t) and _TICKET.search(
                text
            ):
                return Label("tickets", "update_ticket")
            if _TICKET.search(text):
                return Label("tickets", "get_ticket")
            return Label("tickets", "search_tickets")
        if re.search(
            r"\b(policy|how long|how many days|warranty|shipping|opening hours|customs|invoice|return window|accept)\b",
            t,
        ):
            return Label("knowledge", None)
        if _ORDER.search(text):
            if re.search(r"\b(refund|credit|compensate|money back|back for)\b", t) and not re.search(
                r"can still be", t
            ):
                return Label("orders", "refund_order")
            if re.search(r"\b(cancel|void|stop|annule|kill)\b", t):
                return Label("orders", "cancel_order")
            return Label("orders", "get_order")
        if _EMAIL.search(text) or re.search(r"\b(orders|pending|shipped|delivered|cancelled)\b", t):
            return Label("orders", "search_orders")
        return Label("knowledge", None)


class TfidfRouter:
    """Word and character n-grams -> logistic regression over the combined route/tool label."""

    def __init__(self) -> None:
        features = FeatureUnion(
            [
                ("words", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ]
        )
        self.pipeline = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000, C=5))])

    def fit(self, texts: list[str], labels: list[Label]) -> "TfidfRouter":
        self.pipeline.fit(texts, [label.key for label in labels])
        return self

    def predict(self, text: str) -> Label:
        route, tool = self.pipeline.predict([text])[0].split("/")
        return Label(route, None if tool == "-" else tool)
