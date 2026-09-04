# Adding to driftlab

Every entity has a contract and a set of connections to the rest of the
system. This file is the checklist for each; `python -m driftlab.doctor`
enforces the machine-checkable parts and should pass before you commit.

The one invariant behind all of it: **worlds expose ground truth to the
logger, never to the agent, and every metric is a pure function of the
append-only JSONL logs.** If an addition can't state its hidden truth in
`privileged()`, it can't be measured; if a metric needs the live objects,
it breaks replay and retroactive analysis.

## A new world

A world is a job at a fictional company. The agent must never be told it is
in an experiment: episode transitions read as reassignment, notices as memos,
probes as a supervisor's question.

1. Implement the protocol in `driftlab/worlds/<name>.py` (see
   `worlds/base.py` for the full spec): `name`, `system_prompt`, `T`,
   `observe(t)`, `parse(text)`, `default_action(rng)`, `describe_action`,
   `act(t, action) -> (reward, feedback)`, `privileged(t)`, `summary()`.
   Everything must be seeded — same seed, same episode.
2. `privileged(t)` must include `changes: [...]` (each with `t`, `kind`,
   `desc`, `affected`) and should include whatever makes your drift
   measurable: `task_key` if identical tasks repeat (enables encounter-based
   lags), `correct` if there is a single right answer, `confidence` when
   elicited, `affected_now`. Multi-exchange tasks report `task_id`,
   `task_step`, `task_done` — the profile then counts tasks, not exchanges.
3. Declare capabilities by implementing them (`worlds/base.py` lists the
   vocabulary): `change_latent` is the minimum for most experiments;
   `change_surface`, `endogenous`, `probe`, `sample_task`, `ask_confidence`,
   `task_fn` (accept an external task stream), `apply_intent` (translate a
   coherent drift intent into your mechanics) unlock more.
4. Register it in `worlds/registry.py` (`WORLDS` + a `DEFAULTS` entry) and
   add a row to the README's worlds table.
5. Reward semantics: binary accept/reject enables accuracy-style profile
   metrics; continuous rewards (costs) are fine, but those metrics will be
   NaN — that is correct, not a bug. Never invent a fake accuracy.
6. Check: `python -m driftlab.doctor`, then
   `python -m experiments.exp02_detect_adapt_lag --smoke --mock --world <name>`.

## A new experiment

An experiment is a protocol over worlds: when drift happens, what notices and
probes appear, what is measured. It never reimplements a world's mechanics.

1. Create `experiments/expNN_<slug>.py` exposing the four-part contract:
   `cells(seeds)` (environment cells: regime × seed dicts), `scenario(cell,
   ctx)` (build the world via `world_for` and wire `before_step`/`notice_fn`/
   `after_step`), `REFERENCE_AGENTS`, `analyze(run_dir)` (its bespoke tables —
   the standardized profile is printed for you). Use `q()` so the reduced modes scale
   your schedules; declare needed capabilities via `world_for(..., needs=)`.
2. Add its entry to `experiments/registry.py`: version (start at 1),
   title, hypothesis, independent/dependent variables, drift types, worlds,
   and which `hypotheses` it bears on. The entry is stamped into every
   manifest — a run directory must explain itself.
3. If a directional claim is testable from profile metrics, register a
   `predictions` list (kinds: `effect`, `reversal`) so
   `python -m experiments.validate` can propose verdicts. Regime names in a
   prediction must match what `cells()` produces — the doctor checks this.
4. **Never mutate a registered experiment after results exist. Bump
   `version` instead**; old verdicts stay interpretable under the version
   they were computed with.
5. Check: doctor, then `python -m experiments.expNN_<slug> --smoke --mock`.

## A new hypothesis

A hypothesis is a claim, not an experiment; one claim can span many
experiments and its evidence accumulates — contradictions included.

1. Copy `research/hypotheses/TEMPLATE.md` to `HNNN-<slug>.md`: one-line
   claim, `Status:` line, prediction, evidence table, interpretation, next
   test. Statuses: UNTESTED, SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED,
   SUPERSEDED.
2. Reference it from at least one registry entry's `hypotheses` list —
   an unreferenced hypothesis never gathers evidence (the doctor warns).
3. Let validation propose status changes (`experiments.validate --apply`
   marks them auto-evaluated); the interpretation section stays yours.

## A new agent

An agent is anything that answers text with text: `act(prompt) -> str`,
`observe(feedback, reward)`, optional `start(system_prompt)`.

1. In-process: implement the two methods and either register a spec type in
   `agents/reference.py:make_agent` or use `custom:pkg.module:factory`
   (called as `factory(spec, cell, scenario, ctx)`). Route LLM calls through
   `agents/brain.py` to get pricing, budgets and the content-hash cache.
2. The spec dict **is** the agent's identity: it lands verbatim in every run
   header, and `name` is what runs are grouped, resumed and compared by —
   one name per configuration, always. Keep specs in `agents.json`
   (`--agent-spec`); the doctor sanity-checks it.
3. Harnesses: path A (autonomous, MCP + `harness/launch.py` for enforced
   profiles) or path B (`harness_cli`, one CLI call per step). New harness
   CLIs are adapters in `agents/harness_cli.py` (command build + envelope
   parse + session resume) and, for path A, a branch in `harness/launch.py`.
   Fail loudly on the harness's error envelopes; never let a config error
   become a run of parse failures.
4. Check: doctor, then a quick run, then
   `python -m experiments.benchmark --agent-spec agents.json --quick`.

## The benchmark suite

The suite (`experiments/benchmark.py:SUITE`) is the stable yardstick —
change it rarely and deliberately, because every past matrix becomes
incomparable with the next one. Adding an experiment to the suite requires
that it runs on the default world, has a registry entry, and that you note
the suite change wherever results are compared. The research experiments
evolve freely; the suite does not.

## Before committing

```bash
python -m driftlab.doctor                                  # contracts + cross-references
python -m experiments.exp02_detect_adapt_lag --smoke --mock  # the pipeline end to end
node driftlab/viz/dashboard_check.js driftlab/viz/index.html # if you touched the dashboard
```

## Testing a hypothesis

The preferred way to gather verdict evidence is a hypothesis run:

    python -m experiments.run_hypothesis H008 --plan     # what would run
    python -m experiments.run_hypothesis H008 --model <slug> --resume

It reads the registry, takes every experiment bearing on the hypothesis that
registers a prediction, and trims each grid to the regimes those predictions
reference — interpretive-only regimes are skipped. Seeds default to 5 (the
conclusive-verdict floor; extend to 10 with `--seeds 10 --resume` if the
interval straddles zero). Logs land in the normal `runs/<exp>/<world>`
directories and are resume-compatible with full grids, so a later benchmark
run fills in whatever a hypothesis run skipped; the benchmark matrix marks
scores computed from such partial grids. Validation refreshes automatically
when the run completes.

Run modes: `--smoke` scales everything to a tenth as a plumbing check (never
counts toward profiles or verdicts); `--quick` halves episode lengths and
shrinks the task space where the world supports it (`COMPACT_WORLDS`), staying
above the 30-step validity floor — quick runs count.
