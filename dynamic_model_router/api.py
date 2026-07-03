from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .router import Router

app = FastAPI(title="Dynamic Model Router", version="0.1.0")
router = Router()


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    provider: str
    category: str
    confidence: float
    reason: str


@app.post("/route", response_model=RouteResponse)
def route_request(payload: RouteRequest) -> RouteResponse:
    decision = router.route(payload.prompt)
    return RouteResponse(
        provider=decision.selected_provider,
        category=decision.category,
        confidence=decision.confidence,
        reason=decision.reason,
    )
