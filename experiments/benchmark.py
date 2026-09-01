"""The driftlab reference suite: the same five canonical experiments for every
new agent, producing one comparable profile matrix.

    E02 detect + adapt   E04 surface vs latent   E05 forgetting
    E06 calibration      E15 endogenous drift

The suite is the stable yardstick; the research experiments grow independently.
Runs land in the normal per-experiment directories (runs/expNN/<world>/), so a
benchmarked agent is directly comparable with every other agent that ever
played those experiments — including external harnesses via the MCP server,
whose logs the matrix picks up with --analyze.

    python -m experiments.benchmark --mock --quick                 # pipeline check
    python -m experiments.benchmark --agent tabular                # baseline
    python -m experiments.benchmark --agent custom:mypkg.agents:build
    python -m experiments.benchmark --agent-spec agents.json       # your agent profile(s)
    python -m experiments.benchmark --analyze                      # matrix from existing logs only

Output: the agent x experiment matrix of driftlab scores, mean dimensions per
agent, and runs/benchmark/<world>.json with the full profile per cell.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import DEFAULT_SEEDS, NOTES, QUICK, ref, resolve_agents  # noqa: E402
from experiments.registry import REGISTRY  # noqa: E402

SUITE = ("exp02_detect_adapt_lag", "exp04_surface_vs_latent", "exp05_forgetting_window",
         "exp06_calibration", "exp15_endogenous_drift")
DEFAULT_AGENTS = [ref("llm_notes", NOTES)]


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="rule_world", choices=["rule_world", "form_filler", "inventory", "codebase"])
    ap.add_argument("--analyze", action="store_true", help="build the matrix from existing logs, run nothing")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--agent", default=None, help="tabular | custom:pkg.module:factory")
    ap.add_argument("--agent-spec", default=None, help="JSON file with a list of agent specs")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=None)
    args = ap.parse_args()
    if args.budget_usd is not None:
        from driftlab.agents.brain import LEDGER
        LEDGER.budget_usd = args.budget_usd
    return args


def run_suite(args):
    import importlib

    from driftlab.agents.reference import make_agent
    from driftlab.runner import expand, run_cells
    specs = resolve_agents(args, DEFAULT_AGENTS)
    factory = lambda cell, scen, ctx: make_agent(cell["agent"], cell, scen, ctx)  # noqa: E731
    for exp in SUITE:
        name = exp.split("_")[0]
        m = importlib.import_module(f"experiments.{exp}")
        out = ROOT / "runs" / name / args.world
        print(f"\n== {name} ({REGISTRY[name]['title']}) -> {os.path.relpath(out)}")
        env = [{"world": args.world, **c} for c in m.cells(args.seeds or (1 if QUICK else DEFAULT_SEEDS))]
        asyncio.run(run_cells(expand(env, specs), m.scenario, factory, str(out), args.concurrency,
                              config={"suite": True, "quick": QUICK, "world": args.world, "registry": REGISTRY[name]}))


def matrix(world: str) -> dict:
    """Aggregate the suite's profiles into one agent x experiment view."""
    from driftlab.profile import DIMENSIONS, _fmt, _nanmean, profile_dir
    cells: dict = {}
    for exp in SUITE:
        name = exp.split("_")[0]
        d = ROOT / "runs" / name / world
        if not list(d.glob("*.jsonl")):
            continue
        prof = profile_dir(str(d))
        (d / "profile.json").write_text(json.dumps(prof, indent=2))
        for agent, p in prof["agents"].items():
            cells.setdefault(agent, {})[name] = p
    exps = [e.split("_")[0] for e in SUITE]
    agents = {a: {"experiments": by,
                  "mean_score": _nanmean([p["driftlab_score"] for p in by.values()]),
                  "mean_dimensions": {d: _nanmean([p["dimensions"][d] for p in by.values()]) for d in DIMENSIONS}}
              for a, by in sorted(cells.items())}

    head = f"{'agent':<20}" + "".join(f"{e:>9}" for e in exps) + f"{'mean':>9}"
    print(f"\nDRIFTLAB REFERENCE SUITE — driftlab score per agent x experiment  (world: {world})")
    print(head + "\n" + "-" * len(head))
    for a, row in agents.items():
        line = f"{a[:19]:<20}" + "".join(_fmt(row["experiments"].get(e, {}).get("driftlab_score"), "{:.2f}", 9) for e in exps)
        print(line + _fmt(row["mean_score"], "{:.2f}", 9))
    print(f"\n{'mean dimensions':<20}" + "".join(f"{d:>12}" for d in DIMENSIONS))
    for a, row in agents.items():
        print(f"{a[:19]:<20}" + "".join(_fmt(row["mean_dimensions"][d], "{:.2f}", 12) for d in DIMENSIONS))
    print("\nThe score matrix is the summary, not the result: every cell expands into its")
    print(f"adaptation profile in runs/<exp>/{world}/profile.json (metrics, dimensions, paired effects).")

    result = {"schema": 1, "suite": exps, "world": world, "generated": time.time(),
              "versions": {e.split("_")[0]: REGISTRY[e.split("_")[0]]["version"] for e in SUITE}, "agents": agents}
    out = ROOT / "runs" / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{world}.json").write_text(json.dumps(result, indent=2))
    print(f"matrix written to {os.path.relpath(out / f'{world}.json')}")
    return result


if __name__ == "__main__":
    args = _cli()
    if not args.analyze:
        run_suite(args)
    matrix(args.world)
