"""Grid runner: cells x agents -> one JSONL trajectory per run, plus a manifest
with cost. The runner knows nothing about how agents work; it gets an agent
from `agent_factory(cell, scenario, ctx)` and drives it through the scenario.
Harness runs (external agents via MCP) use `write_run` directly."""

import asyncio
import json
import time
from pathlib import Path

from .agents.brain import LEDGER, CostLedger
from .core import run
from .live import LIVE, current_run


def expand(env_cells: list[dict], agents: list[dict]) -> list[dict]:
    """Cross environment-side cells (regime, seed, ...) with agent specs."""
    return [{**c, "agent": a} for c in env_cells for a in agents]


def env_cells(regimes, seeds: int, **extra) -> list[dict]:
    return [{"regime": r, "seed": s, **extra} for r in regimes for s in range(seeds)]


def default_run_id(cell: dict) -> str:
    return f"{cell['agent']['name']}__{cell.get('regime', 'na')}__seed{cell['seed']}"


def write_run(out_dir: Path, run_id: str, cell: dict, header_extra: dict, records: list[dict],
              cost: dict, wall_s: float, config: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"kind": "header", "cell": cell, "ts": time.time(), "wall_s": wall_s,
                            "cost": cost, **header_extra}) + "\n")
        for r in records:
            f.write(json.dumps({"kind": "step", **r}) + "\n")
    entry = {"run": run_id, "path": str(path), "cost": cost, "wall_s": wall_s}
    manifest = out_dir / "manifest.json"
    prior = json.loads(manifest.read_text()) if manifest.exists() else {"runs": [], "grid_costs": []}
    prior["runs"].append(entry)
    if config is not None:
        prior["config"] = config
    prior["unpriced_models"] = sorted(LEDGER.unpriced_models)
    manifest.write_text(json.dumps(prior, indent=2))
    return entry


async def run_cells(cells: list[dict], scenario_factory, agent_factory, out_dir: str, concurrency: int = 6,
                    run_id=default_run_id, config: dict | None = None) -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = {"out_dir": out, "cache_dir": out.parent / "llm_cache", "agent_factory": agent_factory}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def guarded(cell):
        async with sem:
            rid = run_id(cell)
            current_run.set(rid)
            scenario = scenario_factory(cell, ctx)
            agent = agent_factory(cell, scenario, ctx)
            # out is runs/<exp>/<world>; the parent is the experiment (out.name is the world dir)
            exp = out.parent.name if out.parent.name.startswith("exp") else out.name
            LIVE.run_start(cell, scenario.steps, scenario.instructions, experiment=exp,
                           world=getattr(scenario.world, "name", None))
            async with lock:
                before = LEDGER.snapshot()
            t0 = time.time()
            header_extra, records = await run(scenario, agent)
            async with lock:
                cost = CostLedger.diff(LEDGER.snapshot(), before)
            spent = sum(d["cost_usd"] for d in cost.values())
            rewards = [r["reward"] for r in records]
            LIVE.run_end({"steps": len(records), "mean_reward": round(sum(rewards) / len(rewards), 3),
                          "wall_s": round(time.time() - t0, 1), "cost_usd": round(spent, 4),
                          "parse_failures": header_extra.get("parse_failures", 0)})
            return write_run(out, rid, cell, header_extra, records, cost, time.time() - t0, config)

    start = LEDGER.snapshot()
    results = await asyncio.gather(*(guarded(c) for c in cells))
    grid = CostLedger.diff(LEDGER.snapshot(), start)
    spent = sum(d["cost_usd"] for d in grid.values())
    saved = sum(d["cache_saved_usd"] for d in grid.values())
    print(f"grid cost: ${spent:.4f} spent, ${saved:.4f} served from cache")
    return results
