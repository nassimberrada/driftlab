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

    def __init__(self, window: int = 40, wipe_at: int | None = None):
        self.events = deque(maxlen=window)
        self.wipe_at = wipe_at

    def observe(self, t, observation, action, reward, feedback):
        if self.wipe_at is not None and t == self.wipe_at:
            self.events.clear()
            from ..live import LIVE
            LIVE.event(f"context wiped at step {t}: the transcript is gone", kind="notice")
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

    def __init__(self, every: int = 10, budget_chars: int | None = None, trigger: str = "interval",
                 wipe_at: int | None = None):
        """trigger: interval (every `every` steps) | failure (right after a failed step) |
        both | never (notes are only ever loaded, never rewritten). wipe_at: step at which
        raw, unconsolidated history is erased — the written notes survive, which is the point."""
        self.notes = ""
        self.pending = []
        self.every = every
        self.budget_chars = budget_chars
        self.trigger = trigger
        self.wipe_at = wipe_at

    def observe(self, t, observation, action, reward, feedback):
        if self.wipe_at is not None and t == self.wipe_at:
            self.pending = []
            from ..live import LIVE
            LIVE.event(f"context wiped at step {t}: unconsolidated history is gone, the notes survive", kind="notice")
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


class Beliefs(Notes):
    name = "beliefs"
    SYSTEM = ("You maintain a belief file for an agent operating in an environment. Each line is one belief: "
              "`BELIEF: <what you currently believe> | CONFIDENCE: <low|medium|high> | WOULD CHANGE IF: <evidence>`. "
              "Update the file from the new experiences: raise or lower confidence with the evidence, rewrite "
              "beliefs the evidence contradicts, and delete beliefs that no longer apply. Output only belief lines.")

    def recall(self, observation) -> str:
        return f"Your current beliefs:\n{self.notes}" if self.notes else ""


class FastSlow(Substrate):
    """Two timescales at once: a short raw transcript (fast) plus periodically
    rewritten notes (slow). The prompt carries both."""

    name = "fastslow"

    def __init__(self, window: int = 10, every: int = 10, budget_chars: int | None = None):
        self.fast = Transcript(window)
        self.slow = Notes(every=every, budget_chars=budget_chars)

    def observe(self, t, observation, action, reward, feedback):
        self.fast.observe(t, observation, action, reward, feedback)
        self.slow.observe(t, observation, action, reward, feedback)

    def recall(self, observation) -> str:
        parts = [p for p in (self.slow.recall(observation), self.fast.recall(observation)) if p]
        return "\n\n".join(parts)

    async def consolidate(self, brain, t, failed: bool = False):
        await self.slow.consolidate(brain, t, failed)

    def export(self) -> str:
        return self.slow.export()

    def load(self, text):
        self.slow.load(text)


def make_substrate(spec: dict) -> Substrate:
    kind = spec["kind"]
    kwargs = {k: v for k, v in spec.items() if k not in ("kind", "name")}
    return {"none": NoMemory, "transcript": Transcript, "notes": Notes, "skills": Skills,
            "beliefs": Beliefs, "fastslow": FastSlow}[kind](**kwargs)
