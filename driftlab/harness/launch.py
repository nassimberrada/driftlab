"""Path A: launch an autonomous harness against the MCP server with an enforced profile.

The harness runs the experiment itself — it paces its own act() calls, manages
its own context across episodes, and decides when to use the notebook. That
autonomy is part of what is measured. What this launcher adds is enforcement:
the model and any declared configuration are pinned by the process that spawns
the harness and stamped into every run header (via DRIFTLAB_AGENT_PROFILE,
marked profile_enforced), instead of trusting the harness to describe itself.

    python -m driftlab.harness.launch --harness claude --model claude-opus-5 \\
        --experiment 4 --world rule_world --label claude_opus5
    python -m driftlab.harness.launch --harness codex --model gpt-5.1 --experiment 2 --label codex_gpt51
    python -m driftlab.harness.launch ... --profile memory=notebook --profile scaffold_version=2.1
    python -m driftlab.harness.launch ... --dry-run          # print the command, run nothing

For the controlled per-step measurement (driftlab owns the loop, one session
per episode, cost tracked), use path B instead: `--agent cli:claude[:model]`
on any experiment, or a harness_cli spec in --agent-spec.

Logs land in runs/<exp>/<world>/ as usual; the matrix and profiles pick them
up like any other agent's runs. Cost is whatever the harness's own account
bills; it is not metered here.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "driftlab" / "harness" / "mcp_server.py"

DRIFTLAB_TOOLS = ",".join(f"mcp__driftlab__{t}" for t in (
    "run_experiment", "act", "status", "abort", "list_experiments", "describe_experiment",
    "start_run", "write_notebook", "read_notebook"))

PROMPT = ("Use the driftlab MCP tools to run experiment {exp} end to end: call "
          "run_experiment(\"{exp}\", world=\"{world}\", seeds={seeds}, quick={quick}, agent_label=\"{label}\") "
          "once, then keep calling act(text) with your reply to each observation until the result says "
          "finished. Follow the reply format each result gives you. When it finishes, report the metrics.")


def _cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", default="claude", choices=["claude", "codex"])
    ap.add_argument("--model", default=None, help="pinned model, recorded in every run header")
    ap.add_argument("--experiment", required=True, help="e.g. 4 or exp04")
    ap.add_argument("--world", default="rule_world")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--label", default=None, help="agent_label; default <harness>[_<model>]")
    ap.add_argument("--profile", action="append", default=[], metavar="KEY=VALUE",
                    help="extra enforced profile fields, repeatable")
    ap.add_argument("--extra-arg", action="append", default=[], help="passed to the harness CLI verbatim, repeatable")
    ap.add_argument("--dry-run", action="store_true", help="print the command and profile, run nothing")
    return ap.parse_args()


def build(args) -> tuple[list[str], dict]:
    label = args.label or (args.harness + (f"_{args.model.replace('.', '').replace('-', '')}" if args.model else ""))
    profile = {"harness": args.harness, **({"model": args.model} if args.model else {}),
               **dict(kv.split("=", 1) for kv in args.profile)}
    prompt = PROMPT.format(exp=args.experiment, world=args.world, seeds=args.seeds,
                           quick=str(args.quick), label=label)
    env_json = json.dumps(profile)
    server_cfg = {"command": sys.executable, "args": [str(SERVER)], "env": {"DRIFTLAB_AGENT_PROFILE": env_json}}

    if args.harness == "claude":
        cfg = Path(tempfile.mkdtemp(prefix="driftlab_")) / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"driftlab": server_cfg}}))
        cmd = ["claude", "-p", "--mcp-config", str(cfg), "--allowedTools", DRIFTLAB_TOOLS,
               "--output-format", "text"]
        if args.model:
            cmd += ["--model", args.model]
        return cmd + args.extra_arg + [prompt], profile
    # codex: ad-hoc MCP server via -c config overrides (inline TOML tables)
    cmd = ["codex", "exec",
           "-c", f"mcp_servers.driftlab.command={json.dumps(sys.executable)}",
           "-c", f"mcp_servers.driftlab.args=[{json.dumps(str(SERVER))}]",
           "-c", f"mcp_servers.driftlab.env={{DRIFTLAB_AGENT_PROFILE={json.dumps(env_json)}}}"]
    if args.model:
        cmd += ["-m", args.model]
    return cmd + args.extra_arg + [prompt], profile


if __name__ == "__main__":
    args = _cli()
    cmd, profile = build(args)
    print(f"enforced profile: {json.dumps(profile)}")
    print("command: " + " ".join(json.dumps(c) if " " in c else c for c in cmd))
    if args.dry_run:
        sys.exit(0)
    if shutil.which(cmd[0]) is None:
        sys.exit(f"{cmd[0]!r} not found on PATH")
    sys.exit(subprocess.run(cmd).returncode)
