"""HTTP service for Deskpilot's ROUTER_MODE=remote: POST /route {"text": ...} -> {"route", "tool"}.

uvicorn --factory routelab.serve:create_app --port 8700
"""

import os
import threading

from fastapi import FastAPI
from pydantic import BaseModel, Field

from routelab.lora import DEFAULT_BASE, LLMRouter


class RouteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def create_app(router=None) -> FastAPI:  # noqa: ANN001
    app = FastAPI(title="routelab router")
    state: dict = {"router": router}
    lock = threading.Lock()  # one generation at a time: torch on a small CPU gains nothing from more

    def get_router():  # noqa: ANN202
        if state["router"] is None:
            state["router"] = LLMRouter(
                os.environ.get("ROUTER_BASE", DEFAULT_BASE),
                os.environ.get("ROUTER_ADAPTER", "adapter"),
                threads=int(os.environ.get("ROUTER_THREADS", "0")) or None,
            )
        return state["router"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/route")
    def route(body: RouteRequest) -> dict:
        with lock:
            label = get_router().predict(body.text)
        if label is None:  # never happened in evaluation, but the caller must always get a route
            return {"route": "knowledge", "tool": None, "valid": False}
        return {"route": label.route, "tool": label.tool, "valid": True}

    return app
