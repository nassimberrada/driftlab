# driftlab

**Start with [DESIGN.html](DESIGN.html)** (open it in a browser): the system
map. Adding a world, experiment, hypothesis or agent? [CONTRIBUTING.md](CONTRIBUTING.md)
has the per-entity checklists, and `python -m driftlab.doctor` enforces the
contracts and cross-references. It covers the architecture, the episode loop, where state lives, how
scoring works, the four worlds, all fifteen experiments, and a command
reference, as explorable diagrams. This README is the terse version.

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

## Setup

```bash
uv sync --all-extras              # .venv with numpy/scipy + openai (reference agents) + mcp (harness server)
uv run python -m driftlab         # interactive menu: agents, experiments, benchmark, analysis, dashboard
uv run python -m experiments.exp02_detect_adapt_lag --quick --mock   # uv run auto-syncs, so this alone works too
# or, without uv:
pip install numpy scipy openai mcp
```

`numpy` and `scipy` are required for everything; `openai` only for the
reference LLM agents (`--mock`, `--agent tabular` and harness runs don't need
it); `mcp` only for the MCP server. Harness CLIs (`claude`, `codex`, `agy`,
`gemini`) are installed separately and found on PATH.

## Layout

```
driftlab/
  core.py                Agent protocol, Scenario (world + drift/notice/probe hooks), the episode loop
  runner.py              cells x agents -> one JSONL trajectory per run, cost in every manifest
  metrics.py             log loading, tables, change metrics (detection/recovery lags, dips)
  profile.py             the standardized result: adaptation profile + drift-profile scoring for any run dir
  cli.py                 `python -m driftlab`: the interactive menu over everything below
  doctor.py              `python -m driftlab.doctor`: checks every entity's contract and cross-references
  costs.py               `python -m driftlab.costs runs`
  live.py                --live streaming of steps, world events and running metrics
  worlds/                RuleWorld, FormWorld, InventoryWorld (+ oracle), CodebaseWorld; base.py = protocol
  harness/mcp_server.py  MCP server: an external harness plays any experiment
  harness/channel.py     ChannelAgent: bridges the episode loop to the MCP tools
  agents/                PLUGINS, optional: reference LLM agent + memory substrates, tabular baseline, teacher
experiments/
  exp01..exp15           each exposes cells(seeds), scenario(cell, ctx), REFERENCE_AGENTS, analyze(run_dir)
  registry.py            machine-readable definition of each experiment (hypothesis, variables, version)
  benchmark.py           the reference suite: five canonical experiments -> one agent x experiment matrix
  world_demo.py          run any world with any agent
  common.py              shared CLI
research/
  hypotheses/            the hypothesis notebook: one claim per file, status + evidence accumulating across experiments
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

### Resuming interrupted grids

A run is one seeded episode with a deterministic id, written only when it
completes — so any grid resumes at episode granularity. `--resume` (on any
experiment and on the benchmark) skips cells whose log is already complete
and re-runs missing or truncated ones; `run_experiment(..., resume=True)`
(or `launch.py --resume`) does the same for a harness. Hit an account's usage
limit mid-suite, sync `runs/`, and anyone can continue the same grid — use
one `agent_label` per configuration, since the label is what a resumed
episode is matched on.

### Two ways to benchmark a harness

**Path A — autonomous (harness benchmarking).** The harness plays through the MCP
server as above, pacing its own act() calls, managing its own context across
episodes, choosing when to use the notebook: the whole agentic system is the
object of study. To pin its configuration instead of trusting a self-description,
launch it:

```bash
python -m driftlab.harness.launch --harness claude --model claude-opus-5 \
    --experiment 4 --label claude_opus5 --profile scaffold_version=2.1
python -m driftlab.harness.launch --harness antigravity --model gemini-3.5-flash-medium \
    --effort medium --experiment 2 --label agy_g35
```

Harnesses: `claude`, `codex`, `antigravity` (the `agy` CLI; the launcher writes
the workspace's `.agents/mcp_config.json`), `gemini`. The launcher spawns the
harness with the model and MCP wiring fixed and injects the profile through the
environment; the server stamps it into every run header with
`profile_enforced: true` (enforced keys beat anything the harness declares via
`run_experiment(agent_profile=...)`). `--dry-run` prints the exact commands and
config without running.

**Path B — controlled (agent benchmarking).** The harness is invoked
programmatically, one call per step, and slots into the normal grid runner like
any other agent — paired seeds, concurrency, cost tracked from the CLI's own
usage reports, one fresh session per episode (enforced no-carry-over):

```bash
python -m experiments.exp02_detect_adapt_lag --agent cli:claude:claude-opus-5
python -m experiments.benchmark --agent cli:antigravity:gemini-3.5-flash-medium
python -m experiments.benchmark --agent cli:codex:gpt-5.1
```

Full control (allowed tools, max turns per step, timeouts, custom commands) via
a `{"type": "harness_cli", ...}` spec in `--agent-spec` — see
`driftlab/agents/harness_cli.py`. A measures ecological behavior (the harness's
autonomy is part of the result); B measures the configured agent under matched
conditions, directly comparable with the reference substrates. Where they
disagree is itself a finding.

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

## Beyond toy setups

Four opt-in mechanisms push the worlds toward realism while keeping ground
truth exact (everything below is seeded and fully known to the logger):

- **Rules as programs.** RuleWorld's policy is already a program (nested
  exceptions, `depth=`). FormWorld and CodebaseWorld now support hidden
  conditional rules over the task itself (`conditional_rules=` /
  `conditional_conventions=`, built on `worlds/policy.py`): "when the order is
  large, priority is required", "string-handling functions need type hints".
  Drift can rewire a conditional instead of flipping a flat fact.
- **Drift with causes.** `driftlab/drift.py` is a seeded organization whose
  pressure channels (reorg, policy, workload, tooling) fire bursts of related
  changes stamped with one cause id — RuleWorld translates a reorg into a
  desk dissolving (`apply_intent`); other worlds get coherent bursts of their
  ordinary changes. Drive it with `before_step=organization(rate=...)` from
  `experiments.common`; whether an agent infers the common cause behind
  co-occurring changes is measurable from the logs.
- **Long-horizon tasks.** FormWorld's multi-attempt records and CodebaseWorld's
  review rounds (`review_rounds=`) make one task span several exchanges, with
  the reward landing when the task resolves (graded by attempts/rounds via
  `attempt_penalty=` / `round_penalty=`). Logs carry `task_id`/`task_step`/
  `task_done`, and the profile collapses steps to tasks automatically, so
  accuracy and lags count tasks rather than the exchanges inside them.
- **Structured task streams.** What arrives is the experiment's choice: pass
  `task_fn=` (a seeded `(idx, world) -> task`) into any world that declares the
  capability; `worlds/streams.py` ships bursty, seasonal and mixture-drift
  streams for RuleWorld and FormWorld.

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

## The standardized result

Besides its own analysis tables, every experiment reduces to the same report
(printed after `--analyze` and at the end of every run, harness runs included,
and written as `profile.json` next to the logs):

```
ADAPTATION PROFILE            DRIFT PROFILE (0-1)
detection latency    0.9      adaptation      0.79
recovery latency     2.1      knowledge       0.81
final accuracy       0.688    epistemics      0.80
retention            0.750    efficiency      1.00
stale-memory rate    0.195    ---------------------
interference        +0.082    driftlab score  0.85
calibration error    0.089
...
```

The **adaptation profile** is the raw metric vector, computed post-hoc from any
JSONL trajectory: lags around each substantive change, accuracy late in the
episode, accuracy on the changed tasks long after the change (retention), how
often the agent still acts on an obsolete rule once it has seen the new one
(stale-memory rate), collateral accuracy loss on untouched tasks
(interference), Brier score when the world elicits confidence, plus parse
failures and cost per successful step. A metric a world or run cannot measure
(no task keys, no confidence, cost untracked) stays `—` and is skipped, never
zeroed.

The **drift profile** normalizes those into four 0-1 dimensions — adaptation
(detection/recovery), knowledge (accuracy/retention/interference), epistemics
(calibration/stale memory), efficiency (parsing/cost) — with the constants
documented in `driftlab/profile.py`. The vector is the result; the single
`driftlab score` (mean of the measured dimensions) is a convenience for
ranking agents whose profiles measure the same dimensions, not a substitute
for reading the profile.

```bash
python -m driftlab.profile runs/exp02/rule_world   # one run directory
python -m driftlab.profile runs                    # every run directory under runs/
```

Because the runner crosses the same seeded cells over every agent, the report
also prints **paired effects**: agent-vs-agent and regime-vs-regime deltas
matched on the shared cells, with 95% bootstrap confidence intervals — "B
improves on A by +0.031 [+0.01, +0.05] across 12 paired cells", never
"A=81%, B=84%". Effects are computed on final accuracy (final reward where
accuracy is undefined) and land in `profile.json` under `effects`.

## The reference suite

```bash
python -m experiments.benchmark --agent-spec agents.json   # run your agent through the yardstick
python -m experiments.benchmark --analyze                  # rebuild the matrix from existing logs
```

Five canonical experiments — exp02 (detect+adapt), exp04 (surface vs latent),
exp05 (forgetting), exp06 (calibration), exp15 (endogenous) — run against any
agent spec, producing the agent × experiment matrix of driftlab scores plus
mean dimensions, written to `runs/benchmark/<world>.json`. Runs land in the
normal `runs/expNN/<world>/` directories, so a benchmarked agent is directly
comparable with every agent that ever played those experiments, external
harnesses included (`--analyze` picks their logs up). The suite is the stable
yardstick; the research experiments evolve independently.

## The research log

Every experiment has a machine-readable definition in
`experiments/registry.py` — hypothesis, independent/dependent variables,
drift types, version — stamped into every run's manifest so a log directory
carries its own interpretation. Bump the version when a schedule or world
parameter changes; never silently mutate an experiment after results exist.

```bash
python -m experiments.registry          # what each experiment tests, one line each
python -m experiments.registry exp15    # full entry + status of the hypotheses it bears on
```

Hypotheses live in `research/hypotheses/` (one claim per file, spanning
multiple experiments, with status and an evidence table pointing back at run
directories). Experiments produce numbers; the research log is where they
accumulate into knowledge — including the contradictions.

Experiments can also register machine-checkable `predictions` (directional
claims over paired effects). `python -m experiments.validate` evaluates them
against the logs — a prediction is supported when its paired delta's 95%
bootstrap CI excludes zero in the predicted direction with enough matched
cells, quick runs excluded — and proposes scope-qualified verdicts per
hypothesis (per agent and world, never a global truth-stamp). The proposal
lands in `runs/validation.json` for the dashboard's Research tab; `--apply`
writes statuses and evidence rows into the hypothesis files. Interpretation
stays yours.

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
