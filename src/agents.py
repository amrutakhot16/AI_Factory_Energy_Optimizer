"""The 4 AI agents and the orchestrator that runs them.

    1. Usage Analyst    explains how and when the plant uses energy
    2. Waste Detective  explains the unusual spikes
    3. Savings Planner  writes the action plan (JSON)
    4. Reviewer         checks the plan; sends it back once if it's wrong

All numbers come from src/ml.py. The agents only explain them and plan actions.
A code check (check_plan) makes sure the plan never claims more savings than
the data supports. Without an API key, or if Groq fails, each agent falls back
to a simple template so the app always works.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

try:
    from groq import Groq
except ImportError:  # the app still runs offline
    Groq = None

DEFAULT_MODEL = "llama-3.3-70b-versatile"

GROUNDING = ("Use ONLY the numbers given to you. Never invent numbers. Write for a manager "
             "with no engineering background: short sentences, no jargon.")


# ----------------------------------------------------------------------- Groq
class LLM:
    """A thin wrapper around the Groq client with two safety nets:
    it waits and retries on rate limits, and it retries once if Groq's
    strict JSON mode rejects a reply."""

    MAX_WAIT = 45.0  # total seconds to wait for rate limits on one call

    def __init__(self, api_key: str | None, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = Groq(api_key=api_key) if (api_key and Groq) else None

    @property
    def online(self) -> bool:
        return self.client is not None

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if not json_mode:
            return self._call(messages, max_tokens=700)
        try:
            return self._call(messages, max_tokens=1500, response_format={"type": "json_object"})
        except Exception as e:
            if "json_validate_failed" not in str(e):
                raise
            messages[-1]["content"] += "\n\nReply with ONLY the JSON object, no other text."
            return self._call(messages, max_tokens=1500)

    def _call(self, messages: list, max_tokens: int, **kwargs) -> str:
        waited = 0.0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.2, max_tokens=max_tokens, **kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                wait = rate_limit_wait(e)
                if wait is None or waited + wait > self.MAX_WAIT:
                    raise
                print(f"[Groq] Rate limit reached, waiting {wait:.1f}s before retrying...")
                time.sleep(wait)
                waited += wait


def rate_limit_wait(error: Exception) -> float | None:
    """How many seconds Groq asks us to wait, or None if this isn't a rate limit."""
    text = str(error)
    if "rate_limit" not in text and "429" not in text:
        return None
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)(ms|s)", text)
    if not m:
        return 5.0
    hours, minutes, value, unit = m.groups()
    return (float(value) / (1000 if unit == "ms" else 1)
            + 60 * int(minutes or 0) + 3600 * int(hours or 0) + 0.5)


def friendly_error(error: Exception) -> str:
    """A short, safe explanation of a Groq error (never includes keys or account IDs)."""
    text = str(error)
    if "tokens per day" in text or "TPD" in text or "requests per day" in text or "RPD" in text:
        return "Groq's daily limit for this model is used up. Try again later, or set another GROQ_MODEL in .env."
    if "rate_limit" in text or "429" in text:
        return "Groq's per-minute limit was reached and the wait was too long. Try again in a minute."
    if "json_validate_failed" in text or isinstance(error, json.JSONDecodeError):
        return "The AI's reply wasn't valid JSON."
    if "model_not_found" in text or "does not exist" in text or "decommissioned" in text:
        return "The model in GROQ_MODEL isn't available. Pick one from console.groq.com/docs/models."
    if "invalid_api_key" in text or "401" in text:
        return "The GROQ_API_KEY in .env isn't valid."
    if "Connection" in type(error).__name__ or "Timeout" in type(error).__name__:
        return "Couldn't reach Groq (network problem)."
    return f"Unexpected error ({type(error).__name__}). Details are in the Terminal."


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0) if match else text)


def money(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------- Agents
@dataclass
class Agent:
    name: str
    job: str
    system: str                          # the agent's instructions
    build_input: Callable[[dict], str]   # what it reads
    offline: Callable[[dict], object]    # template used without Groq
    json_output: bool = False


# 1. Usage Analyst ------------------------------------------------------------
def _usage_input(s):
    a = s["analysis"]
    return ("Energy summary:\n" + json.dumps(a["summary"], indent=1)
            + "\n\nAverage kWh per 15 minutes, by hour of day:\n" + json.dumps(a["hourly_profile"])
            + "\n\nForecast model quality:\n" + json.dumps(a["forecast"], indent=1))


def _usage_offline(s):
    m, f = s["analysis"]["summary"], s["analysis"]["forecast"]
    return (
        f"- The plant used **{m['total_kwh']:,} kWh**, costing **{money(m['total_cost_usd'])}** "
        f"and producing {m['total_co2_t']} tonnes of CO₂ over {m['days']} days.\n"
        f"- **{m['share_of_energy_in_peak_price_hours']:.0%}** of the energy was used in the expensive "
        f"peak-price hours. The busiest hour is around {m['busiest_hour']}:00.\n"
        f"- Weekdays use about {m['weekday_avg_daily_kwh']:,} kWh a day, weekends about "
        f"{m['weekend_avg_daily_kwh']:,} kWh. {m['share_of_energy_while_plant_closed']:.0%} of all "
        "energy is used at night or on weekends, when the plant is closed.\n"
        f"- The forecast of expected usage is off by about {f['typical_error_kwh_per_15min']} kWh per "
        f"15 minutes, {f['error_reduction_vs_baseline']:.0%} better than guessing 'same as yesterday'."
    )


# 2. Waste Detective ----------------------------------------------------------
def _waste_input(s):
    a = s["analysis"]
    return ("Biggest unusual events (actual vs expected kWh):\n" + json.dumps(a["events"][:8], indent=1)
            + "\n\nRelated savings opportunities:\n" + json.dumps(a["opportunities"][:2], indent=1))


def _waste_offline(s):
    ev = s["analysis"]["events"]
    closed = sum(1 for e in ev if e["when"] == "plant closed")
    lines = [f"- **{len(ev)} unusual events** were found; {closed} of them happened while the plant "
             "was closed, which usually means equipment was left running."]
    for e in ev[:3]:
        lines.append(f"- {e['start']}–{e['end']} ({e['when']}): used {e['actual_kwh']:,} kWh "
                     f"when about {e['expected_kwh']:,} kWh was expected.")
    return "\n".join(lines)


# 3. Savings Planner ----------------------------------------------------------
PLAN_EXAMPLE = json.dumps({"actions": [{
    "title": "Switch off idle equipment at the end of each shift",
    "opportunity": "Switch off machines when the plant is closed",
    "owner": "Maintenance team",
    "timeline": "2-4 weeks",
    "annual_cost_usd": 1000,
    "annual_co2_t": 4.5,
    "priority": "High",
    "why": "One sentence citing the data.",
}]})


def _planner_input(s):
    text = ("Savings opportunities (calculated from the data, per year):\n"
            + json.dumps(s["analysis"]["opportunities"], indent=1)
            + "\n\nUsage Analyst found:\n" + str(s["usage"])
            + "\n\nWaste Detective found:\n" + str(s["waste"])
            + "\n\nReturn a JSON object shaped like this example (values are only illustrative):\n"
            + PLAN_EXAMPLE
            + "\n\nRules: 3-5 actions. Every opportunity above needs at least one action. "
              "Copy \"opportunity\" exactly from the names above. annual_cost_usd and annual_co2_t "
              "are plain numbers. The actions for one opportunity must add up to exactly that "
              "opportunity's annual_cost_usd and annual_co2_t (split it if you use several actions).")
    if "plan" in s:  # this is a revision: pass back the code check's and the Reviewer's problems
        problems = list(s.get("checks") or [])
        problems += [p for p in s.get("review", {}).get("problems") or [] if p not in problems]
        text += ("\n\nYour previous plan was REJECTED. Fix these problems:\n"
                 + json.dumps(problems) + "\nPrevious plan:\n" + json.dumps(s["plan"]))
    return text


def data_backed_plan(s) -> dict:
    """The safe plan, built directly from the numbers. Used without Groq or as a fallback."""
    o = {x["opportunity"]: x for x in s["analysis"]["opportunities"]}
    rows = [
        ("Switch off idle equipment at night and on weekends", "Switch off machines when the plant is closed",
         "Maintenance team", "2-4 weeks", "High"),
        ("Look into the biggest spikes and set up usage alerts", "Investigate unusual spikes",
         "Energy manager", "1-2 weeks", "High"),
        ("Schedule heavy jobs outside peak-price hours", "Run heavy jobs in cheaper hours",
         "Production planning", "1-2 months", "Medium"),
    ]
    return {"actions": [
        {"title": t, "opportunity": name, "owner": owner, "timeline": when,
         "annual_cost_usd": o[name]["annual_cost_usd"], "annual_co2_t": o[name]["annual_co2_t"],
         "priority": prio, "why": o[name]["how_calculated"]}
        for t, name, owner, when, prio in rows
    ]}


# 4. Reviewer -----------------------------------------------------------------
def check_plan(plan: dict, opportunities: list[dict]) -> list[str]:
    """Code check on the AI's plan. Two rules for every savings opportunity:
    1. Nothing left out: at least one action covers it.
    2. Numbers match: its actions add up to the savings the data supports (within 2%),
       so they never claim more, and the plan's total matches the headline figure.
    """
    caps = {o["opportunity"]: o for o in opportunities}
    totals: dict[str, list[float]] = {}
    problems = []
    for a in plan.get("actions", []):
        name = a.get("opportunity")
        cost, co2 = float(a.get("annual_cost_usd") or 0), float(a.get("annual_co2_t") or 0)
        if name in caps:
            t = totals.setdefault(name, [0.0, 0.0])
            t[0] += cost
            t[1] += co2
        elif cost > 0 or co2 > 0:
            problems.append(f"'{a.get('title')}' claims savings but isn't linked to a calculated opportunity.")
    for name, cap in caps.items():
        if cap["annual_cost_usd"] <= 0:
            continue  # nothing to save here, so no action is needed
        if name not in totals:
            problems.append(f"No action covers '{name}' ({money(cap['annual_cost_usd'])}/yr). Add one.")
            continue
        cost, co2 = totals[name]
        if cost > cap["annual_cost_usd"] * 1.02 + 1:
            problems.append(f"'{name}': the plan claims {money(cost)}/yr but the data supports at most "
                            f"{money(cap['annual_cost_usd'])}/yr.")
        elif cost < cap["annual_cost_usd"] * 0.98 - 1:
            problems.append(f"'{name}': the plan only counts {money(cost)}/yr of the "
                            f"{money(cap['annual_cost_usd'])}/yr the data shows.")
        if co2 > cap["annual_co2_t"] * 1.02 + 0.05:
            problems.append(f"'{name}': the plan claims {co2:.1f} t CO₂/yr but the data supports at most "
                            f"{cap['annual_co2_t']} t/yr.")
    if not plan.get("actions"):
        problems.append("The plan has no actions.")
    return problems


def _reviewer_input(s):
    return ("Proposed plan:\n" + json.dumps(s["plan"], indent=1)
            + "\n\nSavings the data supports (each must be covered, and not exceeded):\n" + json.dumps(s["analysis"]["opportunities"], indent=1)
            + "\n\nThe automatic number check found these problems:\n" + json.dumps(s["checks"])
            + '\n\nReturn JSON: {"verdict": "approved" or "needs_revision", "problems": [str], "comment": str}. '
              "Reject if the number check found problems or if an action is vague.")


def _reviewer_offline(s):
    return {"verdict": "needs_revision" if s["checks"] else "approved", "problems": s["checks"],
            "comment": "Number check " + ("failed." if s["checks"] else "passed: every saving matches the data.")}


AGENTS = {
    "usage": Agent("Usage Analyst", "Explains how and when energy is used",
                   "You are an energy analyst. In 4 short bullet points, explain when and how this plant "
                   "uses energy and how reliable the usage forecast is. " + GROUNDING,
                   _usage_input, _usage_offline),
    "waste": Agent("Waste Detective", "Explains the unusual spikes",
                   "You investigate wasted energy. In 4 short bullet points, describe the biggest unusual "
                   "events with their dates, and the likely everyday cause (for example machines left "
                   "running when the plant was closed). " + GROUNDING,
                   _waste_input, _waste_offline),
    "planner": Agent("Savings Planner", "Writes the action plan",
                     "You turn findings into a practical action plan. Respond with JSON only. " + GROUNDING,
                     _planner_input, data_backed_plan, json_output=True),
    "reviewer": Agent("Reviewer", "Checks the plan against the data",
                      "You are a strict reviewer. Check that the plan is specific and that every saving "
                      "is supported by the data. Respond with JSON only.",
                      _reviewer_input, _reviewer_offline, json_output=True),
}


# Orchestrator
@dataclass
class Orchestrator:
    llm: LLM
    trace: list = field(default_factory=list)

    def _run(self, key: str, state: dict) -> tuple[object, bool]:
        """Run one agent. Returns (output, used_ai)."""
        agent = AGENTS[key]
        prompt = agent.build_input(state)
        start, output, error, used_ai = time.time(), None, None, False
        if self.llm.online:
            try:
                raw = self.llm.chat(agent.system, prompt, json_mode=agent.json_output)
                output = parse_json(raw) if agent.json_output else raw.strip()
                used_ai = True
            except Exception as e:
                print(f"[{agent.name}] Groq error: {type(e).__name__}: {e}")
                error = friendly_error(e)
        if output is None:
            output = agent.offline(state)
        self.trace.append({"agent": agent.name, "job": agent.job,
                           "mode": "AI (Groq)" if used_ai else "template",
                           "seconds": round(time.time() - start, 1),
                           "input": prompt, "output": output, "error": error})
        return output, used_ai

    def run(self, analysis: dict, on_step: Callable[[str], None] = lambda _: None) -> dict:
        state = {"analysis": analysis}

        on_step("Usage Analyst")
        state["usage"], _ = self._run("usage", state)
        on_step("Waste Detective")
        state["waste"], _ = self._run("waste", state)

        for attempt in (1, 2):  # write the plan, review it, allow one revision
            on_step("Savings Planner" + (" (revising)" if attempt == 2 else ""))
            state["plan"], planner_used_ai = self._run("planner", state)
            state["checks"] = check_plan(state["plan"], analysis["opportunities"])

            if self.llm.online and not planner_used_ai:
                # The AI couldn't write a plan, so the data-backed plan is in use.
                state["review"] = {"verdict": "approved" if not state["checks"] else "needs_revision",
                                   "problems": state["checks"],
                                   "comment": "The AI planner was unavailable, so the data-backed plan was used."}
                break

            on_step("Reviewer")
            state["review"], _ = self._run("reviewer", state)
            if state["review"].get("verdict") == "approved" and not state["checks"]:
                break
        else:
            # Rejected twice: use the data-backed plan, and keep the reasons.
            reasons = (state["checks"] or state["review"].get("problems") or [])[:3]
            state["plan"] = data_backed_plan(state)
            state["checks"] = check_plan(state["plan"], analysis["opportunities"])
            state["review"] = {"verdict": "approved", "problems": [], "rejected_because": reasons,
                               "comment": "The AI plan was rejected twice, so the data-backed plan was used."}

        state["summary"] = build_summary(state)
        state["trace"] = self.trace
        return state


def build_summary(state: dict) -> str:
    """The final manager summary, assembled in code from the agents' work."""
    a = state["analysis"]
    actions = "\n".join(
        f"{i}. **{x.get('title')}** ({x.get('owner')}, {x.get('timeline')}): "
        f"{money(float(x.get('annual_cost_usd') or 0))}/yr"
        + (f", {x.get('annual_co2_t')} t CO₂/yr" if float(x.get('annual_co2_t') or 0) else "")
        for i, x in enumerate(state["plan"].get("actions", []), 1))
    return (f"### Potential savings: {money(a['total_annual_savings_usd'])} and "
            f"{a['total_annual_co2_t']} tonnes of CO₂ per year\n\n"
            f"**How the plant uses energy**\n\n{state['usage']}\n\n"
            f"**Where energy is wasted**\n\n{state['waste']}\n\n"
            f"**Action plan**\n\n{actions}")
