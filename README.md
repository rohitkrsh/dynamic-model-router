# dynamic-model-router

Dynamic model routing is a lightweight Python project for selecting the most suitable LLM provider for a user request. The router combines keyword heuristics with a simple semantic similarity layer so it is easy to understand, extend, and use as a portfolio project or starting point for more advanced routing pipelines.

## Why this project exists

This project is inspired by the real problem of multi-agent pipelines where the output of one step must remain faithful for the next. A router that picks the right model for the task can improve cost and latency, while keeping the decision logic transparent and deterministic.

## What it does

The router classifies a prompt into a few useful categories:

- code: best for debugging, refactoring, and implementation work
- reasoning: best for planning and multi-step problem solving
- creative: best for storytelling and writing tasks
- summarization: best for briefs, notes, and transcript summaries
- general: fallback for everyday prompts

It now includes a lightweight semantic similarity path that compares incoming prompts against example prompts for each category.

## Quick start

```bash
python -m pip install -e .[dev,api]
python -m dynamic_model_router.cli "Debug this Python function"
```

Example output:

```text
provider=gpt-4.1
category=code
confidence=0.94
reason=Matched code signals with confidence 0.94.
```

## Run the API locally

```bash
uvicorn dynamic_model_router.api:app --reload
```

## Testing

```bash
python -m pytest -q
```

## Roadmap

- add provider adapters for OpenAI, Anthropic, and Bedrock
- add a configurable embedding backend for stronger semantic routing
- add a simple vector store abstraction for retrieval use cases
- add metrics and logging for observability