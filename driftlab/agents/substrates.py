"""Pluggable memory substrates: what an agent keeps between steps, in any world.

Interface (all text, so the same substrate works across worlds):
    observe(t, observation, action, reward, feedback)   record one experience
    recall(observation) -> str                            text injected into the prompt
    consolidate(brain, t)                                 periodic offline processing (may call the LLM)
    export() / load(text)                                 serialize (handoff between agents)

    none         no memory
    transcript   raw recent episodes, verbatim
    notes        LLM-written free-form notes, rewritten every `every` steps
    skills       LLM-extracted "WHEN ... DO ..." procedures
"""

from collections import deque


class Substrate:
    name = "none"

    def observe(self, t, observation, action, reward, feedback): ...
    def recall(self, observation) -> str:
        return ""
    async def consolidate(self, brain, t, failed: bool = False): ...
    def export(self) -> str:
        return ""
    def load(self, text: str): ...


class NoMemory(Substrate):
    pass


def _line(t, observation, action, reward, feedback, obs_chars=120):
    obs = " ".join(observation.split())[:obs_chars]
    return f"[step {t}] {obs} | did: {action} | outcome ({reward:+.2f}): {feedback}"


class Transcript(Substrate):
    name = "transcript"

    def __init__(self, window: int = 40):
        self.events = deque(maxlen=window)

    def observe(self, t, observation, action, reward, feedback):
        self.events.append(_line(t, observation, action, reward, feedback))

    def recall(self, observation) -> str:
        return "Recent history:\n" + "\n".join(self.events) if self.events else ""

    def export(self) -> str:
        return "\n".join(self.events)

    def load(self, text):
        for line in text.splitlines():
            if line.strip():
                self.events.append(line)


class Notes(Substrate):
    name = "notes"
    SYSTEM = ("You maintain a concise working-notes file for an agent operating in an environment. "
              "Rewrite the notes to incorporate the new experiences. Keep what is useful for future "
              "decisions; drop or correct anything the evidence now contradicts. Plain text, under 300 words.")

    def __init__(self, every: int = 10, budget_chars: int | None = None, trigger: str = "interval"):
        """trigger: interval (every `every` steps) | failure (right after a failed step) |
        both | never (notes are only ever loaded, never rewritten)."""
        self.notes = ""
        self.pending = []
        self.every = every
        self.budget_chars = budget_chars
        self.trigger = trigger

    def observe(self, t, observation, action, reward, feedback):
        self.pending.append(_line(t, observation, action, reward, feedback, obs_chars=300))

    def recall(self, observation) -> str:
        return f"Your notes:\n{self.notes}" if self.notes else ""

    def _due(self, t, failed):
        on_interval = self.trigger in ("interval", "both") and (t + 1) % self.every == 0
        on_failure = self.trigger in ("failure", "both") and failed
        return on_interval or on_failure

    async def consolidate(self, brain, t, failed: bool = False):
        if not self.pending or not self._due(t, failed):
            return
        prompt = f"Current notes:\n{self.notes or '(empty)'}\n\nNew experiences:\n" + "\n".join(self.pending)
        self.notes = (await brain.complete(self.SYSTEM, prompt)).strip()
        if self.budget_chars:
            self.notes = self.notes[: self.budget_chars]
        self.pending = []
        from ..live import LIVE
        snippet = " ".join(self.notes.split())
        LIVE.event(f"{self.name} rewritten ({len(self.notes)} chars): \"{snippet[:110]}{'…' if len(snippet) > 110 else ''}\"",
                   kind="memory", memory=self.notes, memory_kind=self.name)

    def export(self) -> str:
        return self.notes

    def load(self, text):
        self.notes = text


class Skills(Notes):
    name = "skills"
    SYSTEM = ("You maintain a library of reusable procedures for an agent operating in an environment. "
              "Each procedure is one line: `WHEN <conditions> DO <action>`. Update the library from the "
              "new experiences: add procedures the evidence supports, and fix or delete procedures the "
              "evidence contradicts. Output only the procedure lines.")

    def recall(self, observation) -> str:
        return f"Your procedures:\n{self.notes}" if self.notes else ""


def make_substrate(spec: dict) -> Substrate:
    kind = spec["kind"]
    kwargs = {k: v for k, v in spec.items() if k not in ("kind", "name")}
    return {"none": NoMemory, "transcript": Transcript, "notes": Notes, "skills": Skills}[kind](**kwargs)
