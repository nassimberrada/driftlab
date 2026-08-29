"""Shared CLI for experiment entry points.

An experiment defines environment cells, a scenario factory, and an analysis.
The CLI decides which world hosts it and which agent plays:

    --world rule_world|form_filler|inventory|codebase   (default rule_world; an experiment refuses a
                                                         world that lacks a capability it needs)
    (default agent)               the experiment's REFERENCE_AGENTS (in-process LLM + memory plugins)
    --agent tabular               the RuleWorld memorization baseline
    --agent custom:pkg.mod:fn     your own factory(spec, cell, scenario, ctx) -> Agent
    --agent-spec agents.json      a JSON list of agent specs
    (harness)                     run the MCP server and let an external harness play; --analyze reads its logs

Logs go to runs/<exp>/<world>/ so the same experiment on different worlds stays separate.
"""

import argparse
import asyncio
import json
import os as _os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEEDS = 3

# Read at import time because experiments define schedules as module constants.
QUICK = ("--quick" in sys.argv) or bool(_os.environ.get("DRIFTLAB_QUICK"))
QUICK_FACTOR = float(_os.environ.get("DRIFTLAB_QUICK_FACTOR", "0.1"))
WORLD = _os.environ.get("DRIFTLAB_WORLD", "rule_world")
if "--world" in sys.argv:
    WORLD = sys.argv[sys.argv.index("--world") + 1]


def q(n: int, minimum: int = 2) -> int:
    """Scale a step count for quick mode (identity otherwise)."""
    return max(minimum, round(n * QUICK_FACTOR)) if QUICK else n


def qlist(xs: list) -> list:
    return [q(x) for x in xs]


# ---- worlds -------------------------------------------------------------------------

def world_for(cell: dict, T: int, needs: tuple = (), **kw):
    """Build the cell's world (cell['world'] or the CLI's --world) and check capabilities."""
    from driftlab.worlds.base import require
    from driftlab.worlds.registry import make_world
    w = make_world(cell.get("world", WORLD), cell["seed"], T=T, **kw)
    if needs:
        require(w, *needs)
    return w


def scheduled(latent=(), surface=(), novelty=()):
    """A before_step hook that applies world changes at the given steps."""
    latent, surface, novelty = set(latent), set(surface), set(novelty)

    def before(t, w):
        if t in latent:
            w.change_latent(t)
        if t in surface:
            w.change_surface(t)
        if t in novelty:
            w.introduce_novelty(t)
    return before


def spaced(n: int, T: int, warmup: int) -> list[int]:
    """n change steps evenly spaced between warmup and T-5 (what RuleWorld's own schedule did)."""
    import numpy as np
    if n <= 0:
        return []
    return [int(x) for x in np.linspace(warmup, max(T - 5, warmup + 1), n + 1)[:-1]]


# ---- CLI ----------------------------------------------------------------------------

def cli(name: str, description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default=WORLD, choices=["rule_world", "form_filler", "inventory", "codebase"])
    ap.add_argument("--analyze", action="store_true", help="analyze existing logs instead of running")
    ap.add_argument("--mock", action="store_true", help="reference agents use a credential-free mock brain")
    ap.add_argument("--quick", action="store_true", help="sense-check mode: schedules scaled to a tenth, one seed")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--model", default=None, help="override the reference agents' model id (default gpt-5-mini)")
    ap.add_argument("--effort", default=None, help="reasoning effort for reasoning models")
    ap.add_argument("--agent", default=None, help="tabular | custom:pkg.module:factory")
    ap.add_argument("--agent-spec", default=None, help="JSON file with a list of agent specs")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default=None, help="log directory (default runs/<exp>/<world>)")
    ap.add_argument("--budget-usd", type=float, default=None, help="abort once cumulative LLM spend exceeds this")
    ap.add_argument("--live", action="store_true", help="stream steps, world events and running metrics")
    ap.add_argument("--live-every", type=int, default=10)
    if extra:
        extra(ap)
    args = ap.parse_args()
    args.out = args.out or str(ROOT / "runs" / name / args.world)
    if args.budget_usd is not None:
        from driftlab.agents.brain import LEDGER
        LEDGER.budget_usd = args.budget_usd
    if args.live:
        from driftlab.live import LIVE
        LIVE.enabled, LIVE.every = True, args.live_every
        if args.concurrency == ap.get_default("concurrency"):
            args.concurrency = 1
    return args


def resolve_agents(args, reference_agents: list[dict]) -> list[dict]:
    if args.agent_spec:
        specs = json.loads(Path(args.agent_spec).read_text())
    elif args.agent == "tabular":
        specs = [{"name": "tabular", "type": "tabular"}]
    elif args.agent and args.agent.startswith("custom:"):
        specs = [{"name": "custom", "type": "custom", "factory": args.agent[len("custom:"):]}]
    else:
        specs = [dict(a) for a in reference_agents]
    for s in specs:
        s.setdefault("type", "reference")
        if s["type"] == "reference":
            s.setdefault("model", "gpt-5-mini")
            s.setdefault("reasoning_effort", "low")
            if args.mock:
                s["mock"] = True
            if args.model:
                s["model"] = args.model
            if args.effort:
                s["reasoning_effort"] = args.effort
    return specs


def run_experiment(name, doc, cells, scenario, reference_agents, analyze, config=None, extra_args=None):
    from driftlab.agents.reference import make_agent
    from driftlab.runner import expand, run_cells

    args = cli(name, doc, extra_args)
    if args.analyze:
        analyze(args.out)
        return
    env = [{"world": args.world, **c} for c in cells(args.seeds or (1 if QUICK else DEFAULT_SEEDS))]  # a cell may pin its own world
    grid = expand(env, resolve_agents(args, reference_agents))
    factory = lambda cell, scen, ctx: make_agent(cell["agent"], cell, scen, ctx)  # noqa: E731
    asyncio.run(run_cells(grid, scenario, factory, args.out, args.concurrency,
                          config={**(config or {}), "quick": QUICK, "world": args.world}))
    import contextlib
    import io
    from driftlab.live import LIVE
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze(args.out)
    LIVE.analysis(name, buf.getvalue())
    print(buf.getvalue())
    print(f"done -> {args.out}   (dashboard: python -m driftlab.viz.server)")


def ref(name: str, substrate: dict, **extra) -> dict:
    return {"name": name, "type": "reference", "substrate": substrate, **extra}


NOTES = {"kind": "notes", "every": 10}
TRANSCRIPT = {"kind": "transcript", "window": 40}
SKILLS = {"kind": "skills", "every": 10}
NONE = {"kind": "none"}
