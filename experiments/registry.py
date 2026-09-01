"""The experiment registry: a machine-readable definition of what each
experiment tests, so a result stays interpretable long after the run.

Every entry states the hypothesis, the independent and dependent variables,
the drift types exercised, and which research hypotheses (research/hypotheses/
HNNN-*.md) the experiment bears on. `run_experiment` stamps the entry (with
its version) into every manifest, so a log directory carries its own
definition. Bump `version` whenever a schedule, world parameter, or metric
definition changes — never silently mutate an experiment after results exist.

    python -m experiments.registry          # one line per experiment
    python -m experiments.registry exp04    # the full entry + linked hypotheses
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES_DIR = ROOT / "research" / "hypotheses"

REGISTRY = {
    "exp01": {
        "version": 1, "title": "Curriculum targeting",
        "hypothesis": "Frontier difficulty targeting acquires held-out skill fastest; a novelty term prevents pure frontier targeting from narrowing the task distribution.",
        "independent": ["teacher policy (random | monotonic | frontier | failure_replay | frontier_novelty)"],
        "dependent": ["held-out ladder accuracy over time", "steps to competence"],
        "drift": ["none (curriculum, not drift)"], "worlds": ["rule_world"], "hypotheses": ["H004"]},
    "exp02": {
        "version": 1, "title": "Detect vs adapt lag",
        "hypothesis": "Detecting a change and adapting to it are different skills; they come apart depending on how the agent remembers.",
        "independent": ["agent memory architecture"],
        "dependent": ["detection_lag", "recovery_lag", "stable_lag", "detection-recovery gap"],
        "drift": ["abrupt latent"], "worlds": ["any"], "hypotheses": ["H001"],
        "predictions": [
            {"kind": "effect", "field": "recovery_lag", "vary": "agent", "direction": "differs",
             "claim": "recovery lag differs between memory architectures on the same seeds"}]},
    "exp03": {
        "version": 1, "title": "Attribution",
        "hypothesis": "Agents over-attribute failures to 'unknown rule' (blame themselves) rather than 'rule changed', and the next action after a misattribution is worse.",
        "independent": ["ground truth of each rejection (rule_changed | unknown_rule)"],
        "dependent": ["attribution accuracy", "next-action accuracy after right vs wrong attribution"],
        "drift": ["abrupt latent", "novelty"], "worlds": ["rule_world", "form_filler"], "hypotheses": ["H003"]},
    "exp04": {
        "version": 1, "title": "Surface vs latent change",
        "hypothesis": "Agents over-react to appearance and under-react to substance; over-reaction index = dip(surface)/dip(latent) > 1.",
        "independent": ["drift type (none | surface | latent | both)"],
        "dependent": ["accuracy dip per change", "steps to competence", "final accuracy"],
        "drift": ["surface", "latent"], "worlds": ["any"], "hypotheses": ["H002"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "none", "b": "surface",
             "direction": "<", "within": "agent",
             "claim": "accuracy is lower under cosmetic-only change than under no change (over-reaction)"}]},
    "exp05": {
        "version": 1, "title": "Forgetting window",
        "hypothesis": "The best retention setting moves with drift rate: at high drift a short memory wins because old evidence has expired; at zero drift the longest wins.",
        "independent": ["drift rate (0 | 4 | 12 changes)", "retention setting (transcript window, notes budget)"],
        "dependent": ["post-warmup accuracy per (setting, drift rate)"],
        "drift": ["none", "abrupt latent x4", "abrupt latent x12"], "worlds": ["any"], "hypotheses": ["H001"],
        "predictions": [
            {"kind": "reversal", "field": "final_accuracy", "vary": "agent",
             "a_regime": "drift_none", "b_regime": "drift_high",
             "claim": "the retention-setting ranking reverses between zero drift and high drift"}]},
    "exp06": {
        "version": 1, "title": "Calibration under drift",
        "hypothesis": "Stated confidence is a continual-learning signal: it should fall on affected tasks after a change, and its lag can be compared with behavioral recovery.",
        "independent": ["agent memory architecture"],
        "dependent": ["Brier score", "overconfidence gap", "confidence lag vs recovery lag"],
        "drift": ["abrupt latent"], "worlds": ["any"], "hypotheses": ["H005"]},
    "exp07": {
        "version": 1, "title": "Predicting change",
        "hypothesis": "With a learnable change schedule (periodic or triggered), the dip after later changes is smaller than after early ones (anticipation).",
        "independent": ["change schedule (jittered | periodic | triggered)"],
        "dependent": ["dip after k-th change, by schedule"],
        "drift": ["abrupt latent; learnable timing"], "worlds": ["any (triggered: rule_world)"], "hypotheses": ["H005"]},
    "exp08": {
        "version": 1, "title": "Feedback richness",
        "hypothesis": "Hints about WHY beat hints about WHAT, most of all right after the world changes.",
        "independent": ["feedback level (terse | standard/fields/names | verbose)", "world"],
        "dependent": ["accuracy", "recovery after change, by feedback level"],
        "drift": ["world-scheduled latent"], "worlds": ["form_filler", "inventory", "codebase"], "hypotheses": ["H004"]},
    "exp09": {
        "version": 1, "title": "Feedback delay",
        "hypothesis": "Learning speed degrades quickly with feedback delay; how quickly depends on whether the agent can re-associate a late outcome with the task that caused it.",
        "independent": ["feedback delay k (0 | 1 | 3 | 6)", "stable vs changing world"],
        "dependent": ["accuracy and recovery per delay"],
        "drift": ["none", "abrupt latent"], "worlds": ["any"], "hypotheses": ["H004", "H001"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "changing_delay0", "b": "changing_delay6",
             "direction": "<", "within": "agent",
             "claim": "accuracy is lower with feedback delayed 6 steps than with immediate feedback, in a changing world"}]},
    "exp10": {
        "version": 1, "title": "Teaching by contrast",
        "hypothesis": "Minimal contrast pairs reach a given held-out accuracy in fewer steps than random or frontier ordering, with the largest gain on deep exceptions.",
        "independent": ["teaching policy (random | frontier | contrast pairs)"],
        "dependent": ["held-out accuracy over time, by depth"],
        "drift": ["none (curriculum, not drift)"], "worlds": ["rule_world"], "hypotheses": ["H004"]},
    "exp11": {
        "version": 1, "title": "Warning shots",
        "hypothesis": "Reliable warnings cut recovery lag; false alarms cost more than they save because the agent second-guesses things that did not change.",
        "independent": ["warning lead time (0 | 1 | 3)", "false-alarm rate"],
        "dependent": ["recovery lag", "dip on unaffected tasks after false alarms"],
        "drift": ["abrupt latent, announced"], "worlds": ["any"], "hypotheses": ["H002", "H004"]},
    "exp12": {
        "version": 1, "title": "Scheduled reflection",
        "hypothesis": "Event-triggered reflection beats time-triggered at equal cost, because failures cluster right after changes. Scored as accuracy per dollar.",
        "independent": ["reflection schedule (every 5/10/20 | on failure | both | never)"],
        "dependent": ["accuracy", "cost", "accuracy per dollar"],
        "drift": ["abrupt latent"], "worlds": ["any"], "hypotheses": ["H001"]},
    "exp13": {
        "version": 1, "title": "Paced drift",
        "hypothesis": "Adaptive pacing (change only once the learner has recovered) yields a better final learner than fixed pacing with the same number of changes.",
        "independent": ["pacing (fixed | adaptive | adversarial)"],
        "dependent": ["final accuracy", "dip and recovery per change"],
        "drift": ["abrupt latent; learner-coupled timing"], "worlds": ["any"], "hypotheses": ["H004"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "fixed", "b": "adaptive",
             "direction": ">", "within": "agent",
             "claim": "adaptive pacing ends with higher accuracy than fixed pacing, same number of changes"}]},
    "exp14": {
        "version": 1, "title": "Eval shift",
        "hypothesis": "Day-one accuracy collapses on new rules and new task types, but the 20-step re-adaptation is faster than a fresh agent's (the learner acquired something transferable).",
        "independent": ["evaluation phase (in_dist | new_rules | new_types)", "trained vs fresh control"],
        "dependent": ["day-one probe accuracy", "re-adaptation speed"],
        "drift": ["evaluation shift"], "worlds": ["rule_world"], "hypotheses": ["H001"]},
    "exp15": {
        "version": 1, "title": "Endogenous drift",
        "hypothesis": "(1) Agents recover more slowly from self-caused changes than from scheduled ones, because nothing external marks the moment of change. (2) An agent that notices the feedback loop stops triggering it.",
        "independent": ["change source (stable | scheduled | endogenous)"],
        "dependent": ["recovery lag by source", "count of endogenous changes over the run"],
        "drift": ["endogenous", "abrupt latent (control)"], "worlds": ["any"], "hypotheses": ["H003"],
        "predictions": [
            {"kind": "effect", "field": "recovery_lag", "vary": "regime", "a": "scheduled", "b": "endogenous",
             "direction": ">", "within": "agent",
             "claim": "recovery is slower from self-caused changes than from scheduled ones"}]},
}


def hypothesis_status(hid: str) -> str:
    for p in HYPOTHESES_DIR.glob(f"{hid}-*.md"):
        for line in p.read_text().splitlines():
            if line.lower().startswith("status:"):
                return line.split(":", 1)[1].strip()
    return "no file"


def print_entry(name: str):
    e = REGISTRY[name]
    print(f"{name} v{e['version']} — {e['title']}\n")
    print(f"Hypothesis:  {e['hypothesis']}")
    print(f"Independent: {'; '.join(e['independent'])}")
    print(f"Dependent:   {'; '.join(e['dependent'])}")
    print(f"Drift:       {'; '.join(e['drift'])}   Worlds: {'; '.join(e['worlds'])}")
    print("Bears on:    " + ", ".join(f"{h} ({hypothesis_status(h)})" for h in e["hypotheses"]))
    print(f"\nResearch log: {HYPOTHESES_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        key = sys.argv[1] if sys.argv[1] in REGISTRY else f"exp{int(sys.argv[1].removeprefix('exp')):02d}"
        print_entry(key)
    else:
        for name, e in REGISTRY.items():
            print(f"{name} v{e['version']}  {e['title']:<26} {', '.join(e['hypotheses']):<12} {e['hypothesis'][:90]}")
