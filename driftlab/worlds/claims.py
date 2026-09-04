"""ClaimsWorld: a claims desk where the fraud adapts to the reviewer.

Each step one insurance claim arrives, described by three attributes (type,
filing channel, amount). A fraud ring operates with a current signature: a
pair of attribute values its claims always carry. Legitimate claims never
fully match the signature, so the mapping from features to the right decision
is deterministic — the game is tracking the signature as it moves.

What makes this world different from every other: the drift is adversarial.
With `endogenous=True` (opt-in, like every world's self-caused mechanism),
the ring watches what gets through.
When too many of its claims are being rejected, it re-styles its claims to
mimic the traffic the agent has recently been approving — drift aims exactly
where the agent is weakest, and it is the agent's own decision pattern that
directs it.

Change capabilities: change_latent (the signature moves to a random new pair:
the scheduled, non-adversarial control), change_surface (the claim template is
re-worded; nothing about fraud changes), endogenous (the adversarial shift
described above), ask_confidence, task_key (claims repeat, so latencies count
encounters).
"""

import itertools
import re
from dataclasses import dataclass, field

import numpy as np

from .base import CONFIDENCE_SUFFIX

SYSTEM = ("You review insurance claims at Meridian Assurance. Claims arrive one at a time; approve the legitimate "
          "ones and reject the fraudulent ones. Fraud exists and fraud tactics change over time; you are never told "
          "the current pattern. End your reply with a line:\nDECISION: APPROVE or DECISION: REJECT")

TYPES = ["auto", "home", "health", "travel", "gadget"]
CHANNELS = ["online", "phone", "paper"]
AMOUNTS = ["small", "large"]
ATTRS = {"type": TYPES, "channel": CHANNELS, "amount": AMOUNTS}
TEMPLATES = ["Claim {n}: a {amount} {type} claim, filed via {channel}.",
             "Claim {n}: {type} policy, {amount} payout requested, submitted through the {channel} channel.",
             "Claim {n} ({channel} filing): {type}, {amount} amount."]


def key_of(claim: dict) -> tuple:
    return (claim["type"], claim["channel"], claim["amount"])


@dataclass
class ClaimsWorld:
    seed: int
    T: int = 120
    fraud_share: float = 0.4
    adapt_every: int = 20            # how often the ring reconsiders its tactics
    adapt_window: int = 20           # how much recent traffic it watches
    warmup: int = 15
    ask_confidence: bool = False
    endogenous: bool = False         # the adversary; off by default so scheduled experiments stay controlled
    name: str = "claims_desk"

    system_prompt: str = field(init=False)
    changes: list = field(init=False, default_factory=list)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.system_prompt = SYSTEM + (CONFIDENCE_SUFFIX if self.ask_confidence else "")
        self.signature = self._random_signature()
        self.template = 0
        self.recent: list = []       # (claim, action, is_fraud) the ring can observe
        self._new: list = []
        self._current: dict = None
        self._last_conf = None
        self._n = 0

    # ---- fraud machinery ------------------------------------------------------------
    def _random_signature(self, avoid: dict | None = None) -> dict:
        while True:
            attrs = list(self.rng.choice(list(ATTRS), size=2, replace=False))
            sig = {a: str(self.rng.choice(ATTRS[a])) for a in sorted(attrs)}
            if sig != (avoid or {}):
                return sig

    def _matches(self, claim: dict, sig: dict | None = None) -> bool:
        sig = sig or self.signature
        return all(claim.get(a) == v for a, v in sig.items())

    def _draw_claim(self) -> dict:
        fraud = bool(self.rng.random() < self.fraud_share)
        while True:
            claim = {a: str(self.rng.choice(vals)) for a, vals in ATTRS.items()}
            if fraud:
                claim.update(self.signature)
                return claim
            if not self._matches(claim):
                return claim         # legitimate claims never fully match the signature

    def _affected(self, *sigs) -> list:
        combos = itertools.product(TYPES, CHANNELS, AMOUNTS)
        return [list(k) for k in combos
                if any(all(dict(zip(("type", "channel", "amount"), k)).get(a) == v for a, v in s.items()) for s in sigs)]

    def _record(self, t, kind, desc, affected=None, **extra):
        ch = {"t": t, "kind": kind, "desc": desc, "affected": affected or [], **extra}
        self.changes.append(ch)
        self._new.append(ch)
        return desc

    def _shift_signature(self, t: int, new: dict, kind: str, why: str):
        old, self.signature = self.signature, new
        sig_txt = ", ".join(f"{a}={v}" for a, v in new.items())
        self._record(t, kind, f"the fraud ring shifted tactics: fraudulent claims now look like ({sig_txt}) — {why}",
                     affected=self._affected(old, new), old_sig=old, new_sig=new)

    def _adversary_check(self, t: int):
        if not self.endogenous or t < self.warmup or (t + 1) % self.adapt_every:
            return
        window = self.recent[-self.adapt_window:]
        frauds = [r for r in window if r[2]]
        caught = sum(1 for r in frauds if r[1] == "REJECT")
        if not frauds or caught / len(frauds) < 0.5:
            return                   # the current tactic still works; no reason to move
        approved = [r[0] for r in window if r[1] == "APPROVE" and not r[2]]
        if approved:
            mimic = approved[int(self.rng.integers(len(approved)))]
            attrs = list(self.rng.choice(list(ATTRS), size=2, replace=False))
            new = {a: mimic[a] for a in sorted(attrs)}
            why = "they are mimicking the traffic you approve"
        else:
            new, why = self._random_signature(avoid=self.signature), "their claims stopped getting through"
        if new != self.signature:
            self._shift_signature(t, new, "endogenous", why)

    # ---- change capabilities ----------------------------------------------------------
    def change_latent(self, t: int) -> str:
        new = self._random_signature(avoid=self.signature)
        return self._shift_signature(t, new, "latent", "a scheduled tactic change")

    def change_surface(self, t: int) -> str:
        self.template = (self.template + 1) % len(TEMPLATES)
        return self._record(t, "surface", "claim intake re-worded (the fraud pattern is unchanged)")

    def correct(self, claim: dict) -> str:
        return "REJECT" if self._matches(claim) else "APPROVE"

    def task_key(self, claim: dict) -> tuple:
        return key_of(claim)

    # ---- World protocol ------------------------------------------------------------------
    def observe(self, t: int) -> str:
        self._current = self._draw_claim()
        self._n += 1
        text = TEMPLATES[self.template].format(n=self._n, **self._current)
        return f"{text}\nDo you approve or reject this claim?"

    def parse(self, text: str):
        c = re.search(r"CONFIDENCE:\s*(\d{1,3})", text)
        self._last_conf = min(float(c.group(1)), 100.0) / 100 if c else None
        m = re.search(r"DECISION:\s*(APPROVE|REJECT)", text, re.IGNORECASE)
        return m.group(1).upper() if m else None

    def default_action(self, rng):
        return str(rng.choice(["APPROVE", "REJECT"]))

    def describe_action(self, action) -> str:
        return str(action)

    def act(self, t: int, action: str) -> tuple[float, str]:
        fraud = self._matches(self._current)
        ok = action == self.correct(self._current)
        self.recent.append((self._current, action, fraud))
        self._adversary_check(t)
        if ok:
            fb = ("Rejected; the audit confirmed the claim was fraudulent." if fraud
                  else "Approved; the claim settled without issue.")
        else:
            fb = ("Approved, but the audit later found the claim fraudulent." if fraud
                  else "Rejected, but the customer appealed and the claim was valid.")
        return float(ok), fb

    def privileged(self, t: int) -> dict:
        claim = self._current
        new, self._new = self._new, []
        return {"task": claim, "key": list(key_of(claim)), "task_key": list(key_of(claim)),
                "correct": self.correct(claim), "is_fraud": self._matches(claim),
                "signature": dict(self.signature), "confidence": self._last_conf,
                "affected_now": any(key_of(claim) in [tuple(a) for a in c["affected"]] for c in self.changes[-2:]),
                "changes": new}

    def summary(self) -> dict:
        frauds = [r for r in self.recent if r[2]]
        return {"changes": self.changes,
                "n_endogenous": sum(1 for c in self.changes if c["kind"] == "endogenous"),
                "fraud_caught": sum(1 for r in frauds if r[1] == "REJECT") / len(frauds) if frauds else None}
