# H006 — Agents update on surprise, not on the world's dynamics

Status: UNTESTED

## The claim, in plain words

When something surprising happens, what an agent should do depends on how its
world changes: after an abrupt rule flip a big surprise means "revise
everything", under gradual drift it means "adjust a little", and in a noisy but
stable world it means nothing at all. Humans read this difference and react to
the same surprise differently per regime. The claim is that LLM agents do not:
they key their updates to the size of the surprise alone, so they abandon
still-correct behavior when feedback merely lies, and cling to stale behavior
when the rules really moved.

## Prediction

- Over-updating under noise (exp16): in a stable world with misleading
  feedback, agents abandon a still-correct behavior after a wrongly punished
  step at a substantial rate (over-update rate well above 0), even though
  nothing changed.
- The memory-size ranking reverses across dynamics (exp16): a long memory wins
  in the noisy-stable regime (it averages the lies away) and loses under real
  change (it averages the truth away) — the exp05 crossover, restated as an
  update-policy failure.
- Temporal bias (exp16): a longer memory is more *confidently* wrong under
  change — overconfidence rises with transcript length, because more stale
  evidence lowers apparent uncertainty while raising real error.
- A structured belief file (belief / confidence / what would change it) should
  reduce both failure directions relative to free-form notes, if explicit
  revision conditions help the agent condition its updates.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. The metrics to watch: overupdate_rate in noisy_stable (new mirror
image of stale_rate), recovery_lag and stale_rate under abrupt/gradual, and
overconfidence per memory arm.

## Next test

exp16 on rule_world: regimes {abrupt ×4, gradual ×12, noisy_stable 15%} ×
memory arms {transcript 5, transcript 120, notes, beliefs, fast+slow}, shared
seeds, confidence elicited every step. A later, stronger test: predict-then-act
prompting (the agent states its expected outcome before acting), which should
sharpen change detection if the failure is in noticing rather than updating.

Sources: human regime-inference under change-point vs random-walk dynamics
(biorxiv 2025.11.26.690700), temporal bias from stale data in time-varying
model-based RL (arXiv 2604.02260), world-model revision as the L3 capability
(arXiv 2604.22748).
