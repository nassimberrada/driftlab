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
     "harness": "claude" | "codex" | "custom",
     "model": "...",                 # pinned, recorded in the run header
     "allowed_tools": [...],         # claude only; default: no tools allowed
     "max_turns": 8,                 # claude only: agentic turns per step
     "timeout_s": 180,               # per step; a timeout is a parse failure
     "bin": "claude",                # executable override (used by tests)
     "extra_args": [...],            # appended verbatim
     "cmd": ["mytool", "--flag"]}    # harness=custom: stateless command; the
                                     # prompt is appended as the last argument

Sessions: claude resumes one session per episode (--resume), so the harness
keeps its own conversational memory within an episode and never across
episodes — enforced no-carry-over. codex tries `codex exec resume`; custom
commands are stateless and get the system prompt re-sent every call.
Feedback is delivered by prepending it to the next step's prompt (one harness
call per step). Claude reports cost per call, which lands in the ledger and
the manifest like any reference agent's spend.
"""

import asyncio
import json

from .brain import LEDGER


class HarnessCLIAgent:
    def __init__(self, spec: dict):
        self.spec = spec
        self.harness = spec.get("harness", "claude")
        self.session: str | None = None
        self.system = ""
        self.pending: str | None = None

    async def start(self, system_prompt: str):
        self.system, self.session, self.pending = system_prompt, None, None

    async def act(self, prompt: str) -> str:
        text = f"(outcome of your previous action) {self.pending}\n\n{prompt}" if self.pending else prompt
        self.pending = None
        cmd = self._command(text)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
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
            self.session = data.get("session_id", self.session)
            usage = data.get("usage") or {}
            LEDGER.record_costed(self.spec.get("model", "claude-cli"), float(data.get("total_cost_usd") or 0.0),
                                 usage.get("input_tokens", 0), usage.get("output_tokens", 0))
            return data.get("result", "") or ""
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
