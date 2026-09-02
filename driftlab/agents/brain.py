"""LLM brain: a thin, cached, budget-capped wrapper around the OpenAI Responses API,
with a process-wide cost ledger.

Every (model, effort, system, prompt) triple is hashed and cached on disk, so
re-running an experiment with unchanged config costs zero API calls and is
exactly reproducible. Cache hits are counted separately in the ledger (they
cost nothing but are reported as "would-have-cost" so a cold rerun is
predictable).

Default model: gpt-5-mini, reasoning effort "low".
"""

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

DEFAULT_MODEL = "gpt-5-mini"


def _load_dotenv():
    """Read KEY=VALUE lines from the repo's .env into the environment (existing vars win)."""
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# USD per 1M tokens: (input, cached input, output). Reasoning tokens bill as output.
# Edit here when prices change; unknown models are costed at 0 and flagged.
PRICES = {
    "gpt-5": (1.25, 0.125, 10.00),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
}


def price_call(model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> float:
    model = model.split("/")[-1]  # vendor-prefixed slugs (openai/gpt-5-mini) price as the bare model
    key = next((k for k in sorted(PRICES, key=len, reverse=True) if model.startswith(k)), None)
    if key is None:
        return 0.0
    p_in, p_cached, p_out = PRICES[key]
    uncached = max(input_tokens - cached_tokens, 0)
    return (uncached * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1e6


class BudgetExceeded(RuntimeError):
    pass


class CostLedger:
    """Aggregates usage and cost across every Brain in the process."""

    FIELDS = ("calls", "cache_hits", "input_tokens", "cached_tokens", "output_tokens",
              "reasoning_tokens", "cost_usd", "cache_saved_usd")

    def __init__(self):
        self.by_model: dict = {}
        self.budget_usd: float | None = None
        self.unpriced_models: set = set()

    def _bucket(self, model):
        return self.by_model.setdefault(model, {f: 0 for f in self.FIELDS})

    def record(self, model, usage: dict, cached_hit: bool):
        b = self._bucket(model)
        cost = price_call(model, usage["input_tokens"], usage["cached_tokens"], usage["output_tokens"])
        if cost == 0 and usage["input_tokens"] and model not in PRICES:
            self.unpriced_models.add(model)
        if cached_hit:
            b["cache_hits"] += 1
            b["cache_saved_usd"] += cost
            return
        b["calls"] += 1
        for f in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
            b[f] += usage[f]
        b["cost_usd"] += cost
        self._check_budget()

    def record_costed(self, model: str, cost_usd: float, input_tokens: int = 0, output_tokens: int = 0):
        """Record a call whose price is already known (a harness CLI reporting its own spend)."""
        b = self._bucket(model)
        b["calls"] += 1
        b["input_tokens"] += input_tokens
        b["output_tokens"] += output_tokens
        b["cost_usd"] += cost_usd
        self._check_budget()

    def _check_budget(self):
        if self.budget_usd is not None and self.total_cost() > self.budget_usd:
            raise BudgetExceeded(f"spent ${self.total_cost():.2f} > budget ${self.budget_usd:.2f}")

    def total_cost(self) -> float:
        return sum(b["cost_usd"] for b in self.by_model.values())

    def snapshot(self) -> dict:
        return deepcopy(self.by_model)

    @staticmethod
    def diff(after: dict, before: dict) -> dict:
        out = {}
        for model, b in after.items():
            a = before.get(model, {})
            d = {f: b[f] - a.get(f, 0) for f in CostLedger.FIELDS}
            if d["calls"] or d["cache_hits"]:
                out[model] = d
        return out


LEDGER = CostLedger()


class Brain:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str = "runs/llm_cache",
                 max_calls: int = 2000, max_output_tokens: int = 1500, reasoning_effort: str = "low",
                 base_url: str | None = None, api: str | None = None):
        from openai import AsyncOpenAI

        self.model = model
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        self.calls_made = 0
        self.tokens_used = 0
        # Any OpenAI-compatible provider works: set base_url in the agent spec (or
        # OPENAI_BASE_URL in .env) and put that provider's key in OPENAI_API_KEY.
        # OPENROUTER_API_KEY is recognized directly: a vendor-prefixed model slug
        # (openai/gpt-5-mini, anthropic/...) routes to OpenRouter with that key.
        # OpenAI itself speaks the Responses API; other providers get Chat
        # Completions unless the spec forces api="responses".
        effective_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            self._client = AsyncOpenAI(base_url=base_url)
        elif not effective_url and "/" in model and os.environ.get("OPENROUTER_API_KEY"):
            effective_url = "https://openrouter.ai/api/v1"
            self._client = AsyncOpenAI(base_url=effective_url, api_key=os.environ["OPENROUTER_API_KEY"])
        else:
            self._client = AsyncOpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL)
        self.api = api or ("chat" if effective_url and "openai.com" not in effective_url else "responses")

    def _cache_path(self, system: str, prompt: str) -> Path:
        key = hashlib.sha256(f"{self.model}\x00{self.reasoning_effort}\x00{system}\x00{prompt}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage", None)
        g = lambda obj, name: getattr(obj, name, 0) or 0  # noqa: E731
        return {
            "input_tokens": g(u, "input_tokens"),
            "output_tokens": g(u, "output_tokens"),
            "cached_tokens": g(getattr(u, "input_tokens_details", None), "cached_tokens"),
            "reasoning_tokens": g(getattr(u, "output_tokens_details", None), "reasoning_tokens"),
        }

    async def complete(self, system: str, prompt: str) -> str:
        path = self._cache_path(system, prompt)
        if path.exists():
            rec = json.loads(path.read_text())
            LEDGER.record(self.model, rec["usage"], cached_hit=True)
            self.tokens_used += rec["usage"]["input_tokens"] + rec["usage"]["output_tokens"]
            return rec["text"]
        if self.calls_made >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget of {self.max_calls} exhausted")
        self.calls_made += 1
        if self.api == "chat":
            kwargs = dict(model=self.model,
                          messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                          max_completion_tokens=self.max_output_tokens,
                          extra_body={"reasoning": {"effort": self.reasoning_effort}})  # OpenRouter-style; others ignore it
            resp = await self._client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            u = getattr(resp, "usage", None)
            g = lambda obj, name: getattr(obj, name, 0) or 0  # noqa: E731
            usage = {"input_tokens": g(u, "prompt_tokens"), "output_tokens": g(u, "completion_tokens"),
                     "cached_tokens": g(getattr(u, "prompt_tokens_details", None), "cached_tokens"),
                     "reasoning_tokens": g(getattr(u, "completion_tokens_details", None), "reasoning_tokens")}
        else:
            kwargs = dict(model=self.model, instructions=system, input=prompt, max_output_tokens=self.max_output_tokens)
            if self.model.startswith(("gpt-5", "o")):
                kwargs["reasoning"] = {"effort": self.reasoning_effort}
            resp = await self._client.responses.create(**kwargs)
            text = resp.output_text or ""
            usage = self._usage(resp)
        self.tokens_used += usage["input_tokens"] + usage["output_tokens"]
        LEDGER.record(self.model, usage, cached_hit=False)
        path.write_text(json.dumps({"model": self.model, "system": system, "prompt": prompt,
                                    "text": text, "usage": usage}))
        return text


class MockBrain:
    """Credential-free stand-in. `responder(system, prompt) -> str` supplies the
    reply; the default returns "" so agents fall back to their parse-failure
    defaults. Reports estimated tokens to the ledger under model "mock" ($0)."""

    def __init__(self, responder=None, **_):
        self.responder = responder or (lambda s, p: "")
        self.model = "mock"
        self.calls_made = 0
        self.tokens_used = 0

    async def complete(self, system: str, prompt: str) -> str:
        self.calls_made += 1
        text = self.responder(system, prompt)
        usage = {"input_tokens": (len(system) + len(prompt)) // 4, "output_tokens": max(len(text) // 4, 1),
                 "cached_tokens": 0, "reasoning_tokens": 0}
        self.tokens_used += usage["input_tokens"] + usage["output_tokens"]
        LEDGER.record(self.model, usage, cached_hit=False)
        return text


def make_brain(spec: dict, cache_dir: str):
    if spec.get("mock"):
        return MockBrain()
    return Brain(
        model=spec.get("model", DEFAULT_MODEL),
        cache_dir=cache_dir,
        max_calls=spec.get("max_calls", 2000),
        max_output_tokens=spec.get("max_output_tokens", 1500),
        reasoning_effort=spec.get("reasoning_effort", "low"),
        base_url=spec.get("base_url"),
        api=spec.get("api"),
    )
