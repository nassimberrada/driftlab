# driftlab

A testbed for one question: how do agents cope when the world keeps changing
under them? driftlab supplies the changing worlds, the experiments, and the
measurements. It assumes nothing about how an agent remembers or learns — an
agent is anything that answers text with text:

```python
class Agent(Protocol):
    async def act(self, prompt: str) -> str                         # observation in, reply out
    async def observe(self, feedback: str, reward: float) -> None   # outcome of the last action
```

Every world is seeded and knows its own hidden state, so learning is measured
against ground truth rather than a benchmark score. The same measurements
apply whether the agent is an in-process LLM loop, a lookup-table baseline, or
an external coding agent playing through the MCP server.

This file is the short tour. [DESIGN.html](DESIGN.html) is the system map —
how every part works and connects (open it in a browser).
[CONTRIBUTING.md](CONTRIBUTING.md) has the checklists for adding worlds,
experiments, hypotheses and agents. `python -m driftlab.doctor` enforces the
contracts.

## Setup

```bash
uv sync --all-extras                 # numpy/scipy + openai (API agents) + mcp (harness server)
uv run python -m driftlab            # interactive menu over everything
uv run python -m experiments.exp02_detect_adapt_lag --smoke --mock   # end-to-end check, no credentials
```

API keys go in `.env`: `OPENAI_API_KEY` for OpenAI, `OPENROUTER_API_KEY` for
any vendor-prefixed slug (`openai/gpt-5.6-luna`, `anthropic/claude-sonnet-5`),
`OPENAI_BASE_URL` for other compatible providers. Harness CLIs (`claude`,
`codex`, `agy`, `gemini`) are found on PATH — always pin their model.

## The worlds

Six jobs at fictional companies. The agent is never told it is in an
experiment: changes arrive unannounced, notices read as office memos, probes
as a supervisor's question.

| World | The job | What really changes |
|---|---|---|
| rule_world | intake desk: route requests | a hidden routing rule flips |
| form_filler | data entry into an order form | the schema mutates |
| inventory | buyer for one product line | the demand rate jumps |
| codebase | developer implementing tickets | the style guide is revised |
| claims_desk | approve or reject insurance claims | the fraud ring moves against the agent |
| campaign_desk | pick each day's outreach angle | audience tastes are redrawn; feedback only arrives pooled |

Each world can also change cosmetically (only the wording moves — reacting is
the mistake) and, opt-in, endogenously (the agent's own behavior triggers the
change).

## Experiments and hypotheses

Nineteen experiment protocols, each named for the question it asks (does
noticing a change mean adjusting to it? do looks matter more than substance?
does the chase against an adversary ever end?). Each has a versioned registry
entry with its design and, where checkable, a registered prediction:

```bash
python -m experiments.registry           # one line per experiment
python -m experiments.registry exp18     # the full entry
```

Claims live in `research/hypotheses/` (H001–H008), one per file, accumulating
evidence across experiments. The preferred way to test one is a **hypothesis
run** — it runs exactly the regimes the registered predictions need and ends
with a scope-qualified verdict:

```bash
python -m experiments.run_hypothesis H008 --plan     # what would run, and why
python -m experiments.run_hypothesis H008 --model openai/gpt-5.6-luna --budget-usd 15 --resume
```

## Agents

`agents.json` holds the roster; any experiment takes it with
`--agent-spec agents.json`. In-process agents pair a model with a memory
substrate (`none`, `transcript`, `notes`, `skills`, `beliefs`, `fastslow`).
External harnesses play two ways: autonomously over MCP, or invoked once per
step like any other agent. Every LLM call is priced, budgeted and cached.

```bash
python -m experiments.exp02_detect_adapt_lag --agent tabular                       # non-LLM baseline
python -m experiments.exp02_detect_adapt_lag --agent cli:claude:claude-sonnet-5    # a harness, per step
python -m experiments.benchmark --agent-spec agents.json                           # the five-experiment yardstick
```

## Measurement

Every run directory reduces to the same standardized report (`profile.json`):
raw metrics (detection and recovery latency, retention, stale-memory and
over-update rates, calibration, cost per success — unmeasured stays `—`,
never zero), normalized into four 0–1 dimensions whose mean is the driftlab
score, plus **paired effects**: because every agent plays the same seeded
worlds, differences come with 95% bootstrap intervals, never bare percentages.

```bash
python -m driftlab.profile runs/exp02/rule_world
python -m driftlab.viz.server --open        # the dashboard, computed entirely from the logs
```

## Run modes

- **full** — the real thing.
- **`--quick`** — half-length episodes, smaller task space; stays valid and
  **counts**. Confirm anything that matters at full length.
- **`--smoke`** — a tenth of everything, one seed: a plumbing check that
  never counts. `--smoke --mock` needs no credentials.

## Conventions

- Worlds expose ground truth to the logger, never to the agent.
- Everything is seeded: same seed, same episode, byte for byte.
- Metrics are pure functions of the append-only logs; runs are written
  atomically, so anything resumes with `--resume`.
- `--budget-usd` aborts a run past its cap; `python -m driftlab.costs runs`
  reports all spend.
