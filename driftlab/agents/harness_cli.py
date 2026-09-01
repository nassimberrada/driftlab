"""Path B: a harness CLI as a controlled in-process agent.

The harness (Claude Code, Codex, or any command) is invoked programmatically,
one call per step, so it slots into the normal grid runner like any other
agent: cells x agents with paired seeds, concurrency, cost in the manifest.
driftlab owns the loop (pacing, what context arrives when, a fresh session per
episode); the harness owns everything inside a step. This measures the
configured agent under controlled conditions — for the autonomous open-loop
measurement, where the harness paces itself through the MCP server, use
path A: `python -m driftlab.harness.launch`.

Spec (in an --agent-spec JSON list, or --agent cli:claude[:model]):

    {"name": "claude_opus", "type": "harness_cli",
     "harness": "claude" | "codex" | "antigravity" (alias "agy") | "gemini" | "custom",
     "model": "...",                 # pinned, recorded in the run header
     "effort": "low|medium|high",    # antigravity only: reasoning intensity
     "allowed_tools": [...],         # claude only; default: no tools allowed
     "max_turns": 8,                 # claude only: agentic turns per step
     "timeout_s": 180,               # per step; a timeout is a parse failure
     "reply_only": true,             # append a no-tools directive; calls run in an empty temp cwd
     "bin": "claude",                # executable override (used by tests)
     "extra_args": [...],            # appended verbatim
     "cmd": ["mytool", "--flag"]}    # harness=custom: stateless command; the
                                     # prompt is appended as the last argument

Sessions: claude resumes one session per episode (--resume), antigravity via
--conversation <id> from its JSON envelope — either way the harness keeps its
own conversational memory within an episode and never across episodes:
enforced no-carry-over. codex tries `codex exec resume`. gemini continues via
`--resume latest`, which is only safe at --concurrency 1. custom commands are
stateless and get the system prompt re-sent every call. Feedback is delivered
by prepending it to the next step's prompt (one harness call per step).
claude reports cost per call, which lands in the ledger and the manifest;
antigravity reports token usage (recorded, unpriced).
"""

import asyncio
import json
import shutil
import tempfile

from .brain import LEDGER

BINS = {"claude": "claude", "codex": "codex", "antigravity": "agy", "gemini": "gemini"}

# Agentic CLIs given a bare question will reach for their tools — explore the cwd,
# grep for the answer, hang on a permission prompt. Every call therefore runs in an
# empty temp directory (so exploring finds nothing, and never driftlab's own source,
# which contains the worlds' hidden rules), and the instructions end with:
REPLY_ONLY = ("\n\nAnswer each message directly in plain text. Do not use tools, read files, "
              "run commands, or search; just reply to the message.")


class HarnessCLIAgent:
    def __init__(self, spec: dict):
        self.spec = spec
        self.harness = {"agy": "antigravity"}.get(spec.get("harness", "claude"), spec.get("harness", "claude"))
        binary = spec.get("bin") or (spec["cmd"][0] if self.harness == "custom" else BINS.get(self.harness))
        if binary and shutil.which(binary) is None:  # fail fast, not one parse failure per step
            raise SystemExit(f"harness_cli: {binary!r} not found on PATH (agent {spec.get('name')!r}); "
                             f"install the {self.harness} CLI or point \"bin\" at it")
        self.session: str | None = None
        self.system = ""
        self.pending: str | None = None
        self.workdir = tempfile.mkdtemp(prefix="driftlab_cli_")

    async def start(self, system_prompt: str):
        self.system = system_prompt + (REPLY_ONLY if self.spec.get("reply_only", True) else "")
        self.session, self.pending = None, None

    async def act(self, prompt: str) -> str:
        text = f"(outcome of your previous action) {self.pending}\n\n{prompt}" if self.pending else prompt
        self.pending = None
        cmd = self._command(text)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=self.workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.spec.get("timeout_s", 180))
        except (asyncio.TimeoutError, OSError):
            return ""  # unparseable -> the world's default action, counted as a parse failure
        if proc.returncode != 0:
            return ""
        return self._parse(out.decode(errors="replace"))

    async def observe(self, feedback: str, reward: float):
        self.pending = feedback

    # ---- per-harness command building and reply parsing -------------------------------

    def _command(self, text: str) -> list[str]:
        s = self.spec
        if self.harness == "custom":
            base = list(s["cmd"])
            if self.session is None:
                text = f"{self.system}\n\n{text}"  # stateless: instructions travel with the first call
                self.session = "stateless"
            return base + [text]
        if self.harness == "codex":
            cmd = [s.get("bin", "codex"), "exec", "--json"]
            if self.session:
                cmd = [s.get("bin", "codex"), "exec", "resume", self.session, "--json"]
            if s.get("model"):
                cmd += ["-m", s["model"]]
            if self.session is None:
                text = f"{self.system}\n\n{text}"
            return cmd + list(s.get("extra_args", [])) + [text]
        if self.harness == "antigravity":
            if self.session is None:
                text = f"{self.system}\n\n{text}"
            cmd = [s.get("bin", "agy"), "-p", text, "--output-format", "json"]
            if self.session:
                cmd += ["--conversation", self.session]
            if s.get("model"):
                cmd += ["--model", s["model"]]
            if s.get("effort"):
                cmd += ["--effort", s["effort"]]
            return cmd + list(s.get("extra_args", []))
        if self.harness == "gemini":
            if self.session is None:
                text = f"{self.system}\n\n{text}"
            cmd = [s.get("bin", "gemini"), "-p", text, "-o", "json"]
            if self.session:  # `latest` is the only headless continuation; safe only at --concurrency 1
                cmd += ["--resume", "latest"]
            if s.get("model"):
                cmd += ["-m", s["model"]]
            return cmd + list(s.get("extra_args", []))
        # claude
        cmd = [s.get("bin", "claude"), "-p", "--output-format", "json"]
        if self.session:
            cmd += ["--resume", self.session]
        else:
            cmd += ["--append-system-prompt", self.system]
        if s.get("model"):
            cmd += ["--model", s["model"]]
        cmd += ["--allowedTools", ",".join(s.get("allowed_tools", []))]  # default: no tools
        if s.get("max_turns"):
            cmd += ["--max-turns", str(s["max_turns"])]
        return cmd + list(s.get("extra_args", [])) + [text]

    def _parse(self, out: str) -> str:
        if self.harness == "claude":
            try:
                data = json.loads(out)
            except ValueError:
                return out.strip()
            if data.get("is_error"):
                raise SystemExit(f"harness_cli (claude): {data.get('result') or data.get('subtype')}")
            self.session = data.get("session_id", self.session)
            usage = data.get("usage") or {}
            LEDGER.record_costed(self.spec.get("model", "claude-cli"), float(data.get("total_cost_usd") or 0.0),
                                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            return data.get("result", "") or ""
        if self.harness in ("antigravity", "gemini"):
            try:
                data = json.loads(out)
            except ValueError:
                self.session = self.session or "started"
                return out.strip()
            if data.get("status") == "ERROR" or data.get("error"):  # config errors fail every step; die loudly
                raise SystemExit(f"harness_cli ({self.harness}): {data.get('error') or data}")
            self.session = data.get("conversation_id") or data.get("session_id") or self.session or "started"
            usage = data.get("usage") or {}
            if usage.get("input_tokens") or usage.get("output_tokens"):
                LEDGER.record_costed(self.spec.get("model", f"{self.harness}-cli"), float(data.get("total_cost_usd") or 0.0),
                                     usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            reply = next((data[k] for k in ("result", "response", "output", "text", "content")
                          if isinstance(data.get(k), str)), "")
            return reply or out.strip()
        if self.harness == "codex":
            reply = ""
            for line in out.splitlines():  # JSONL events; keep the last agent message, remember the session
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                self.session = ev.get("session_id") or (ev.get("msg") or {}).get("session_id") or self.session
                item = ev.get("item") or {}
                if item.get("type") == "agent_message":
                    reply = item.get("text", reply)
                elif (ev.get("msg") or {}).get("type") == "agent_message":
                    reply = ev["msg"].get("message", reply)
            return reply or out.strip()
        return out.strip()
