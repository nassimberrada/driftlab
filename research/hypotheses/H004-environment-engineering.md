# H004 — The environment can be engineered to optimize the agent's learning

Status: UNTESTED

## Prediction

Holding the agent fixed, the environment side controls learning speed:

- Curricula: frontier-with-novelty task targeting acquires held-out skill
  fastest (exp01); minimal contrast pairs beat random ordering, most on deep
  exceptions (exp10).
- Feedback design: WHY-feedback beats WHAT-feedback, most right after a
  change (exp08); learning degrades quickly with feedback delay (exp09).
- Change design: reliable warnings cut recovery lag (exp11); adaptive pacing
  (change only once the learner recovered) yields a better final learner than
  fixed pacing with the same number of changes (exp13).

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. This is the co-evolving-environment thesis stated as a testable
family: each experiment varies one environment-side lever with the agent held
constant, so the paired regime effects are direct estimates of each lever.

## Next test

exp13 (pacing) is the purest single-lever test: fixed vs adaptive pacing,
one agent, paired seeds.
