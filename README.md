# driftlab

A testbed for studying how agents learn from experience when the world keeps
changing. driftlab configures **environments**, **experiments** and
**metrics**. It makes no assumptions about how an agent remembers, reflects or
learns: an agent is anything that answers text with text.

```python
class Agent(Protocol):
    async def act(self, prompt: str) -> str                      # observation or query in, reply out
    async def observe(self, feedback: str, reward: float) -> None  # outcome of the last action
```

Every environment is seeded and knows its own hidden state, so learning is
measured against ground truth rather than a benchmark score, and the same
metrics apply whether the agent is an in-process LLM loop, a memorization
baseline, or an external harness (Claude Code, Codex, Cursor, a script) playing
through the MCP server.

## Layout

```
driftlab/
  core.py                Agent protocol, Scenario (world + drift/notice/probe hooks), the episode loop
  runner.py              cells x agents -> one JSONL trajectory per run, cost in every manifest
  metrics.py             log loading, tables, change metrics (detection/recovery lags, dips)
  costs.py               `python -m driftlab.costs runs`
  live.py                --live streaming of steps, world events and running metrics
  worlds/                RuleWorld, FormWorld, InventoryWorld (+ oracle), CodebaseWorld; base.py = protocol
  harness/mcp_server.py  MCP server: an external harness plays any experiment
  harness/channel.py     ChannelAgent: bridges the episode loop to the MCP tools
  agents/                PLUGINS, optional: reference LLM agent + memory substrates, tabular baseline, teacher
experiments/
  exp01..exp14           each exposes cells(seeds), scenario(cell, ctx), REFERENCE_AGENTS, analyze(run_dir)
  world_demo.py          run any world with any agent
  common.py              shared CLI
```

## Choosing the agent

```bash
python -m experiments.exp04_surface_vs_latent                  # reference agents (LLM + memory plugin; needs OPENAI_API_KEY)
python -m experiments.exp04_surface_vs_latent --mock           # pipeline check, no credentials
python -m experiments.exp02_detect_adapt_lag --agent tabular   # memorization baseline
python -m experiments.exp02_detect_adapt_lag --agent custom:mypkg.agents:build   # your factory(spec, cell, scenario, ctx)
python -m experiments.exp02_detect_adapt_lag --agent-spec agents.json           # a JSON list of specs
python -m experiments.exp04_surface_vs_latent --analyze        # tables; reads every log in runs/exp04 regardless of who played
```

### An external harness as the agent (MCP)

```bash
claude mcp add driftlab -- python3 /absolute/path/to/driftlab/driftlab/harness/mcp_server.py   # or: codex mcp add ...
```

Then just ask the harness to run an experiment, e.g. "use driftlab to run experiment 4 on the inventory
world in quick mode". It calls `run_experiment("4", world="inventory", quick=True)` once and then
`act(text)` repeatedly. Two extra tools, `write_notebook(text)` and `read_notebook()`, are the harness's
memory: calling them is entirely its own choice (driftlab never requires it), but they are real
tools rather than a file path mentioned in prose, so memory works even for a harness with no
filesystem access of its own. Every `write_notebook` call is versioned in the run log and browsable
on the dashboard's memory panel. Every result carries the outcome of the
previous action, the next observation, and the exact reply format, so the agent never has to know
the internals. The server walks through all of the experiment's environment configurations
(regimes x seeds; one seed by default for harness runs), announces each new episode as a fresh
world, and returns the computed metrics when the last one finishes. Logs land in `runs/expNN/` and
`--analyze` compares them with any other agent's runs. Cost is not tracked for harness runs.

`start_run` / `describe_experiment` give single-configuration control for when you want it.

## Worlds

Every world is a job at a fictional company. The agent is never told it is in an experiment.

| World | The job | Substantive change (`change_latent`) | Cosmetic change (`change_surface`) | Self-caused change (`endogenous`) |
|---|---|---|---|---|
| RuleWorld | intake desk: assign each request to a desk A–D | a routing rule flips | desk descriptions relabeled | a desk the agent overloads sheds a request type to another desk |
| FormWorld | data entry into an order form | field renamed, format/enum/required/cap changed | source-record column names change | IT adds an alias for a field name the agent keeps typing |
| InventoryWorld | buyer for one product line | demand rate jumps | daily report re-worded | a big order strains the supplier; lead time grows for a week |
| CodebaseWorld | developer implementing tickets | style guide revised | ticket template re-worded | reviewer adopts a habit seen in the agent's last three submissions |

Capabilities are declared per world (`driftlab/worlds/base.py`): all four have
`change_latent`, `change_surface`, `ask_confidence`, `endogenous`; RuleWorld and
FormWorld add `introduce_novelty` and `probe`; RuleWorld alone has `task_key`
(identical tasks repeat, enabling encounter-based lags), `sample_task` and
`sample_contrast_pair`. CodebaseWorld executes submitted code in a subprocess with
a timeout; that is a convenience, not a sandbox.

## Worlds x experiments

Experiments schedule changes themselves through the capabilities, so the same
experiment runs on any world that has what it needs. `--world` picks the world
(default `rule_world`); logs go to `runs/<exp>/<world>/`.

| # | Script | Needs | rule_world | form_filler | inventory | codebase |
|---|---|---|---|---|---|---|
| 1 | `exp01_curriculum` | sample_task, probe | ✓ | | | |
| 2 | `exp02_detect_adapt_lag` | change_latent | ✓ | ✓ | ✓ | ✓ |
| 3 | `exp03_attribution` | change_latent, introduce_novelty | ✓ | ✓ | | |
| 4 | `exp04_surface_vs_latent` | change_latent, change_surface | ✓ | ✓ | ✓ | ✓ |
| 5 | `exp05_forgetting_window` | change_latent | ✓ | ✓ | ✓ | ✓ |
| 6 | `exp06_calibration` | change_latent, ask_confidence | ✓ | ✓ | ✓ | ✓ |
| 7 | `exp07_predict_change` | change_latent (triggered: rule_world) | ✓ | ✓ | ✓ | ✓ |
| 8 | `exp08_feedback_richness` | its own three worlds | | ✓ | ✓ | ✓ |
| 9 | `exp09_feedback_delay` | change_latent | ✓ | ✓ | ✓ | ✓ |
| 10 | `exp10_teaching_by_contrast` | sample_task, sample_contrast_pair, probe | ✓ | | | |
| 11 | `exp11_warning_shots` | change_latent | ✓ | ✓ | ✓ | ✓ |
| 12 | `exp12_scheduled_reflection` | change_latent | ✓ | ✓ | ✓ | ✓ |
| 13 | `exp13_paced_drift` | change_latent | ✓ | ✓ | ✓ | ✓ |
| 14 | `exp14_eval_shift` | sample_task, probe, task_key | ✓ | | | |
| 15 | `exp15_endogenous_drift` | change_latent, endogenous | ✓ | ✓ | ✓ | ✓ |

On worlds without `task_key`, lag metrics are time-based (steps until rolling
reward recovers) instead of encounter-based.

## Framing

Every world is presented to the agent as a plain job at a fictional company (an intake desk, a buyer, a
data-entry specialist, a platform developer). Nothing tells the agent it is in an experiment, what is
being measured, or that anything will change; episode transitions read as a reassignment to a
different site and notices read as office memos. Evaluation probes are phrased as a supervisor's
quick check. The experiment's description and cell names are for you (`describe_experiment`, the
docstrings), never for the agent.

## Quick mode

```bash
python -m experiments.exp04_surface_vs_latent --quick --mock      # seconds, no credentials
python -m experiments.exp04_surface_vs_latent --quick             # a few minutes with the reference agent
```

`--quick` (or `DRIFTLAB_QUICK=1`, or `run_experiment(..., quick=True)` from a harness) scales every
step count to a tenth (world length, event schedules, evaluation cadence; `DRIFTLAB_QUICK_FACTOR`
overrides the factor) and defaults to one seed. Experiment 4 becomes four 12-step episodes. It is
for checking that a setup works and produces the right shape of output, not for drawing
conclusions: at this length the metrics are noise.

## Dashboard

```bash
python -m driftlab.viz.server --open          # http://localhost:8765
```

Every run appends structured events to `runs/events.jsonl` (in-process runs and harness runs alike).
The dashboard tails that file and shows, live: the current observation, the agent's reply and the
outcome; world changes, notices and memory rewrites on a timeline; rolling and cumulative accuracy
with event markers; a per-request-key grid for RuleWorld (last outcome per key, changed keys marked);
running metrics and the experiment's final analysis table. Finished runs can be replayed from their
JSONL logs at several speeds. Light and dark themes follow the system.

## Watching, cost, conventions

- `--live` streams every step, world events (rule flips, notices, relabels) and running metrics
  every `--live-every` steps; it sets `--concurrency 1` unless overridden.
- The reference agents' LLM calls are priced from `PRICES` in `agents/brain.py`; every run header
  and manifest carries a cost block; `--budget-usd X` aborts once spend passes X;
  `python -m driftlab.costs runs` reports everything.
- Worlds expose ground truth to the logger, never to the agent. All runs are seeded. LLM calls are
  content-hashed and cached under `runs/llm_cache/`. Metrics are computed post-hoc from logs.
