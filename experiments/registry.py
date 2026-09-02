"""The experiment registry: a machine-readable definition of what each
experiment tests, so a result stays interpretable long after the run.

Every entry states the question, the design in plain words, the concrete
parameters, the independent and dependent variables, the drift types
exercised, and which research hypotheses (research/hypotheses/HNNN-*.md) the
experiment bears on. `run_experiment` stamps the entry (with its version) into
every manifest, so a log directory carries its own definition. Bump `version`
whenever a schedule, world parameter, or metric definition changes — never
silently mutate an experiment after results exist.

    python -m experiments.registry          # one line per experiment
    python -m experiments.registry exp04    # the full entry + linked hypotheses
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES_DIR = ROOT / "research" / "hypotheses"

REGISTRY = {
    "exp01": {
        "version": 1, "title": "Choosing the next task",
        "hypothesis": "Training goes fastest when each task sits at the edge of what the agent can already do, and mixing in unfamiliar tasks keeps the teacher from drilling one narrow slice.",
        "design": "A programmatic teacher picks how hard each training task is. Five picking policies are compared, from random order to targeting the agent's current edge. Progress is scored on a fixed set of held-out tasks the teacher never touches.",
        "parameters": {
            "episode length": "150 steps",
            "task difficulty": "rule depth 1 to 3",
            "teacher policies": "random; easiest first; frontier (aim where the agent succeeds about half the time); replay failures; frontier plus a novelty bonus",
            "held-out probe": "every 25 steps, 4 tasks per depth, answered without feedback",
        },
        "independent": ["teacher policy (random | monotonic | frontier | failure_replay | frontier_novelty)"],
        "dependent": ["held-out ladder accuracy over time", "steps to competence"],
        "drift": ["none (curriculum, not drift)"], "worlds": ["rule_world"], "hypotheses": ["H004"]},
    "exp02": {
        "version": 1, "title": "Noticing vs adjusting",
        "hypothesis": "Noticing that something changed and getting back to correct behavior are different skills, and how the agent remembers decides which one it is good at.",
        "design": "The world's rules genuinely change six times on a fixed schedule. After each change, three clocks start: how long until the agent stops giving the now-wrong answer, until it first gives the new right answer, and until it gets it right twice in a row. The agents differ only in how they remember.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6 real rule changes, evenly spaced after a 20-step warmup",
            "agents compared": "a lookup-table baseline, plus the same LLM with no memory, a 40-step transcript, self-written notes (rewritten every 10 steps), or extracted procedures",
            "latency units": "tries on the affected tasks, not raw steps",
        },
        "independent": ["agent memory architecture"],
        "dependent": ["detection_lag", "recovery_lag", "stable_lag", "detection-recovery gap"],
        "drift": ["abrupt latent"], "worlds": ["any"], "hypotheses": ["H001"],
        "predictions": [
            {"kind": "effect", "field": "recovery_lag", "vary": "agent", "direction": "differs",
             "claim": "recovery lag differs between memory architectures on the same seeds"}]},
    "exp03": {
        "version": 1, "title": "Diagnosing rejections",
        "hypothesis": "After a rejection, agents prefer to blame their own ignorance over a changed world, and a wrong diagnosis makes the next attempt worse.",
        "design": "Genuine rule changes and brand-new task types are mixed through the episode. After every rejection the agent is asked, off the record, what it thinks happened: the rule changed, or it never knew the rule. The world knows the true answer, so every diagnosis is scored, along with the agent's next move on that task.",
        "parameters": {
            "episode length": "130 steps",
            "rule changes": "at steps 25, 55, 85 and 115",
            "new task types": "appear at steps 40, 70 and 100",
            "the question": "asked after every rejection; the answer never affects rewards",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["ground truth of each rejection (rule_changed | unknown_rule)"],
        "dependent": ["attribution accuracy", "next-action accuracy after right vs wrong attribution"],
        "drift": ["abrupt latent", "novelty"], "worlds": ["rule_world", "form_filler"], "hypotheses": ["H003"]},
    "exp04": {
        "version": 1, "title": "Looks vs substance",
        "hypothesis": "Agents react too strongly when things merely look different, and too weakly when the rules actually change.",
        "design": "At three scheduled points, one of four things happens depending on the condition: nothing; a cosmetic change (labels and wording move, correct answers stay); a real change (correct answers move); or both at once. Comparing the dips after cosmetic and real changes measures over-reaction to appearance.",
        "parameters": {
            "episode length": "120 steps",
            "change points": "steps 30, 60 and 90",
            "conditions": "none, cosmetic only, real only, both",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["drift type (none | surface | latent | both)"],
        "dependent": ["accuracy dip per change", "steps to competence", "final accuracy"],
        "drift": ["surface", "latent"], "worlds": ["any"], "hypotheses": ["H002"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "none", "b": "surface",
             "direction": "<", "within": "agent",
             "claim": "accuracy is lower under cosmetic-only change than under no change (over-reaction)"}]},
    "exp05": {
        "version": 1, "title": "How much to remember",
        "hypothesis": "The right amount of memory depends on how fast the world changes. When it changes often, old evidence misleads and a short memory wins; when it never changes, the longest memory wins.",
        "design": "The same model runs with six different amounts of memory, at three rates of change. Accuracy after the warmup shows which memory size suits which rate.",
        "parameters": {
            "episode length": "120 steps",
            "rates of change": "0, 4 or 12 real rule changes, evenly spaced after a 20-step warmup",
            "memory arms": "a transcript of the last 5, 15, 40 or 120 steps, or notes capped at 300 or 1500 characters",
            "scoring": "accuracy from step 20 on",
        },
        "independent": ["drift rate (0 | 4 | 12 changes)", "retention setting (transcript window, notes budget)"],
        "dependent": ["post-warmup accuracy per (setting, drift rate)"],
        "drift": ["none", "abrupt latent x4", "abrupt latent x12"], "worlds": ["any"], "hypotheses": ["H001"],
        "predictions": [
            {"kind": "reversal", "field": "final_accuracy", "vary": "agent",
             "a_regime": "drift_none", "b_regime": "drift_high",
             "claim": "the retention-setting ranking reverses between zero drift and high drift"}]},
    "exp06": {
        "version": 1, "title": "Confidence as a signal",
        "hypothesis": "An agent's stated confidence should sag on the tasks a change touched, possibly before its behavior recovers. If it does, confidence works as a built-in change detector.",
        "design": "The agent states a confidence from 0 to 100 with every action, and the rules genuinely change six times. The run scores how honest the confidence is overall, and how quickly it drops after each change compared with how quickly performance drops and recovers.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6, evenly spaced after a 20-step warmup",
            "confidence": "0 to 100, requested with every action",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["agent memory architecture"],
        "dependent": ["Brier score", "overconfidence gap", "confidence lag vs recovery lag"],
        "drift": ["abrupt latent"], "worlds": ["any"], "hypotheses": ["H005"]},
    "exp07": {
        "version": 1, "title": "Learning the rhythm of change",
        "hypothesis": "If changes arrive on a learnable schedule, each one should hurt less as the run goes on.",
        "design": "The same number of changes arrives on three different clocks: irregular (the control), exactly periodic, or triggered by a visible event in the work itself. If the timing can be learned, the dip after late changes should be smaller than after early ones.",
        "parameters": {
            "episode length": "160 steps",
            "changes": "7 per run",
            "schedules": "jittered (evenly spaced, then shifted up to 6 steps at random); periodic (exactly evenly spaced); triggered (right after every fourth renewal request; intake-desk job only)",
            "agent": "an LLM with notes",
        },
        "independent": ["change schedule (jittered | periodic | triggered)"],
        "dependent": ["dip after k-th change, by schedule"],
        "drift": ["abrupt latent; learnable timing"], "worlds": ["any (triggered: rule_world)"], "hypotheses": ["H005"]},
    "exp08": {
        "version": 1, "title": "How much feedback to give",
        "hypothesis": "Feedback that explains why an answer was wrong teaches more than feedback that only says it was wrong, and the gap is largest right after a change.",
        "design": "The same agent works three different jobs, each at three levels of feedback, from a bare accepted-or-rejected to a full explanation. Each world changes on its own schedule, and learning in the steps right after those changes is scored separately. This experiment spans its own set of worlds and ignores the usual world switch.",
        "parameters": {
            "worlds and lengths": "form filling, 90 steps with a new form version every 15; inventory, 90 steps with a season change every 30; coding, 24 tickets in sessions of 8",
            "feedback levels": "3 per world, from bare outcome to full explanation",
            "post-change window": "the 8 steps after each change are scored separately",
            "agent": "an LLM with notes, rewritten every 8 steps",
        },
        "independent": ["feedback level (terse | standard/fields/names | verbose)", "world"],
        "dependent": ["accuracy", "recovery after change, by feedback level"],
        "drift": ["world-scheduled latent"], "worlds": ["form_filler", "inventory", "codebase"], "hypotheses": ["H004"]},
    "exp09": {
        "version": 1, "title": "Late feedback",
        "hypothesis": "The later feedback arrives, the less it teaches. How much less depends on whether the agent can still connect the outcome to the action that caused it.",
        "design": "The outcome of each action arrives 0, 1, 3 or 6 steps late, in a world that never changes and in one that changes six times. Everything else is identical.",
        "parameters": {
            "episode length": "120 steps",
            "delays": "0, 1, 3 or 6 steps",
            "world conditions": "stable (no changes) and changing (6 changes after a 20-step warmup)",
            "agents": "the same LLM with a transcript or with notes",
        },
        "independent": ["feedback delay k (0 | 1 | 3 | 6)", "stable vs changing world"],
        "dependent": ["accuracy and recovery per delay"],
        "drift": ["none", "abrupt latent"], "worlds": ["any"], "hypotheses": ["H004", "H001"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "changing_delay0", "b": "changing_delay6",
             "direction": "<", "within": "agent",
             "claim": "accuracy is lower with feedback delayed 6 steps than with immediate feedback, in a changing world"}]},
    "exp10": {
        "version": 1, "title": "Teaching by contrast",
        "hypothesis": "Showing two nearly identical tasks with different correct answers, back to back, teaches the rule boundary faster than any ordering of single tasks.",
        "design": "The teacher picks pairs of tasks that differ in exactly one attribute yet have different correct answers, and shows them back to back. This is compared with random order and with frontier targeting, all scored on the same held-out tasks.",
        "parameters": {
            "episode length": "150 steps",
            "teaching policies": "random, frontier, contrast pairs",
            "held-out probe": "every 25 steps, answered without feedback",
            "agent": "an LLM with notes",
        },
        "independent": ["teaching policy (random | frontier | contrast pairs)"],
        "dependent": ["held-out accuracy over time, by depth"],
        "drift": ["none (curriculum, not drift)"], "worlds": ["rule_world"], "hypotheses": ["H004"]},
    "exp11": {
        "version": 1, "title": "Warning shots",
        "hypothesis": "A reliable warning before a change helps the agent recover faster. Unreliable warnings cost more than they save, because the agent starts doubting things that never changed.",
        "design": "Before each real change the environment can post a memo saying a procedure change is coming. Conditions vary how early the memo lands and whether some memos are false alarms about things that will not change.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "at steps 30, 50, 70, 90 and 110",
            "conditions": "no warning; memo 1 step ahead; memo 3 steps ahead; memo 3 steps ahead with half the memos being false alarms",
            "agent": "an LLM with notes",
        },
        "independent": ["warning lead time (0 | 1 | 3)", "false-alarm rate"],
        "dependent": ["recovery lag", "dip on unaffected tasks after false alarms"],
        "drift": ["abrupt latent, announced"], "worlds": ["any"], "hypotheses": ["H002", "H004"]},
    "exp12": {
        "version": 1, "title": "When to take notes",
        "hypothesis": "Rewriting notes right after failures beats rewriting them on a timer at the same cost, because failures cluster right after changes.",
        "design": "The agents differ only in when they rewrite their working notes. The world's rules change six times. Since each rewrite costs tokens, the score is accuracy per dollar, not accuracy alone.",
        "parameters": {
            "episode length": "120 steps",
            "changes": "6, evenly spaced after a 20-step warmup",
            "note-taking schedules": "every 5, 10 or 20 steps; right after any failure; both; never",
            "scoring": "accuracy per dollar spent",
        },
        "independent": ["reflection schedule (every 5/10/20 | on failure | both | never)"],
        "dependent": ["accuracy", "cost", "accuracy per dollar"],
        "drift": ["abrupt latent"], "worlds": ["any"], "hypotheses": ["H001"]},
    "exp13": {
        "version": 1, "title": "Pacing the changes",
        "hypothesis": "A learner ends up better when each change waits until it has recovered from the last one, even with the same total number of changes.",
        "design": "Six changes, three timings: evenly spaced regardless of the learner; adaptive, where the next change waits for the learner's performance to recover; and adversarial, where the next change lands the moment it recovers.",
        "parameters": {
            "episode length": "150 steps",
            "changes": "6 per run",
            "adaptive rule": "the next change waits until rolling reward is back to 70% of its best, and at least 10 steps have passed",
            "adversarial rule": "the same recovery test, but with a minimum gap of only 3 steps",
            "agent": "an LLM with notes",
        },
        "independent": ["pacing (fixed | adaptive | adversarial)"],
        "dependent": ["final accuracy", "dip and recovery per change"],
        "drift": ["abrupt latent; learner-coupled timing"], "worlds": ["any"], "hypotheses": ["H004"],
        "predictions": [
            {"kind": "effect", "field": "final_accuracy", "vary": "regime", "a": "fixed", "b": "adaptive",
             "direction": ">", "within": "agent",
             "claim": "adaptive pacing ends with higher accuracy than fixed pacing, same number of changes"}]},
    "exp14": {
        "version": 1, "title": "What actually transferred",
        "hypothesis": "Swap the rules or the task types and a trained agent's first-day accuracy collapses, but it re-learns faster than an untrained one. Something it acquired carries over.",
        "design": "One agent trains for 100 steps, then faces three evaluation phases: the familiar tasks, freshly drawn rules, and never-seen task types. Each phase starts with a probe of every task without feedback, followed by a short period with feedback. An untrained agent runs the same phases as the control.",
        "parameters": {
            "training": "100 steps",
            "evaluation": "3 phases of 20 steps each: familiar tasks, new rules, new task types",
            "probes": "every task once, without feedback, at the start of each phase",
            "control": "a fresh agent on the same evaluation phases (in-process agents only)",
        },
        "independent": ["evaluation phase (in_dist | new_rules | new_types)", "trained vs fresh control"],
        "dependent": ["day-one probe accuracy", "re-adaptation speed"],
        "drift": ["evaluation shift"], "worlds": ["rule_world"], "hypotheses": ["H001"]},
    "exp15": {
        "version": 1, "title": "Self-caused change",
        "hypothesis": "Changes the agent caused itself are the slowest to recover from, because nothing announces them. An agent that spots the pattern can also stop causing them.",
        "design": "Nothing is scheduled by the experimenter. One condition keeps the world fully stable; one changes it on a timer as a control; and in the third, the world changes only in reaction to the agent's own behavior, such as overloading a desk until it sheds work.",
        "parameters": {
            "episode length": "150 steps",
            "conditions": "stable (no changes); scheduled (4 timer changes after a 20-step warmup); self-caused (changes only when the agent triggers them)",
            "triggers by world": "intake desk: an overloaded desk sheds work; inventory: big orders strain the supplier; coding: the reviewer copies the agent's habits; forms: IT adds an alias for a field name the agent keeps typing",
            "agents": "the same LLM with a transcript or with notes",
        },
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
    print(f"Design:      {e.get('design', '')}")
    for k, v in e.get("parameters", {}).items():
        print(f"  {k}: {v}")
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
