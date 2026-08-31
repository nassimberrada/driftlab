# H001 — Memory architecture sets the terms of adaptation

Status: UNTESTED

## Prediction

How an agent remembers determines the split between detecting a change and
recovering from it, and the trade between retention and staleness:

- Detection and recovery lags come apart by substrate (exp02): raw transcripts
  detect fast but recover slowly; consolidated notes detect slower but recover
  in fewer encounters once they do.
- The best retention setting moves with drift rate (exp05): short windows win
  at high drift, long windows at zero drift — the crossover is measurable.
- Event-triggered reflection beats time-triggered at equal cost (exp12).
- Whatever is acquired transfers: a trained agent re-adapts to fresh rules
  faster than a fresh agent, even when day-one accuracy collapses (exp14).
- Under delayed feedback the memory benefit shrinks or reverses (exp09),
  because stale associations get written before outcomes arrive.

## Evidence

| Experiment | World | Runs | Verdict | Notes |
|---|---|---|---|---|
| — | — | — | — | no full-scale runs yet |

## Current interpretation

None yet. The profile metrics to watch: detection_lag vs recovery_lag gap,
retention, stale_rate, and the paired memory-vs-no-memory effect per regime.

## Next test

exp02 + exp05 on rule_world, reference substrates x 10+ shared seeds; read the
paired effects block for substrate-vs-substrate deltas per drift rate.
