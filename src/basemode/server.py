"""OpenAI-completions-compatible HTTP server around basemode.

Exposes POST /v1/completions so tools built for llama.cpp's llama-server or
the OpenAI legacy completions API (e.g. Tapestry Loom) can drive basemode's
model-coerced continuations instead of a locally-hosted base model.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .continue_ import branch_text, continue_text
from .keys import get_default_model

app = FastAPI(title="basemode completions server")


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str] = ""
    max_tokens: int = 200
    temperature: float = 0.9
    n: int = 1
    echo: bool = False
    strategy: str | None = None


def _prompt_to_text(prompt: str | list[str]) -> str:
    if isinstance(prompt, list):
        return "".join(p for p in prompt if isinstance(p, str))
    return prompt


def _choice(index: int, prefix: str, text: str, echo: bool) -> dict[str, Any]:
    return {
        "index": index,
        "text": (prefix + text) if echo else text,
        "logprobs": None,
        "finish_reason": "stop",
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    default_model = get_default_model() or "gpt-4o-mini"
    return {
        "object": "list",
        "data": [{"id": default_model, "object": "model", "owned_by": "basemode"}],
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> dict[str, Any]:
    prefix = _prompt_to_text(request.prompt)
    model = request.model or get_default_model() or "gpt-4o-mini"

    try:
        if request.n <= 1:
            text = "".join(
                [
                    token
                    async for token in continue_text(
                        prefix,
                        model=model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        strategy=request.strategy,
                        strict_max_tokens=True,
                    )
                ]
            )
            choices = [_choice(0, prefix, text, request.echo)]
        else:
            texts = ["" for _ in range(request.n)]
            async for idx, token in branch_text(
                prefix,
                model=model,
                n=request.n,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                strategy=request.strategy,
                strict_max_tokens=True,
            ):
                texts[idx] += token
            choices = [
                _choice(i, prefix, text, request.echo) for i, text in enumerate(texts)
            ]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
