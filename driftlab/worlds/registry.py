"""World registry: build any world by name, and report what it can do."""

from .base import capabilities
from .codebase import CodebaseWorld
from .formfiller import FormWorld
from .gridtext import RuleWorld
from .inventory import InventoryWorld

WORLDS = {"rule_world": RuleWorld, "form_filler": FormWorld, "inventory": InventoryWorld, "codebase": CodebaseWorld}

# reasonable per-world defaults for a "standard" episode, used when an experiment does not care
DEFAULTS = {
    "rule_world": {"T": 120, "depth": 2},
    "form_filler": {"T": 90, "version_every": 10_000},   # changes are scheduled by scenarios
    "inventory": {"T": 90, "regime": "stationary"},
    "codebase": {"T": 24, "session_length": 10_000},
}


def make_world(name: str, seed: int, **kw):
    cls = WORLDS[name]
    params = {**DEFAULTS.get(name, {}), **kw}
    accepted = set(cls.__dataclass_fields__)
    return cls(seed=seed, **{k: v for k, v in params.items() if k in accepted})


def capability_table() -> dict:
    return {name: sorted(capabilities(cls(seed=0))) for name, cls in WORLDS.items()}
