"""Check your Groq setup: tests the key and model, then runs the Savings Planner alone.
"""
import os

import pandas as pd
from dotenv import load_dotenv

from src.agents import DEFAULT_MODEL, LLM, Orchestrator, _usage_offline, _waste_offline, friendly_error
from src.ml import Tariff, analyze, load

load_dotenv()
key = os.getenv("GROQ_API_KEY", "").strip()
model = os.getenv("GROQ_MODEL", "").strip() or DEFAULT_MODEL
print("Model:  ", model)
print("API key:", "found" if key else "MISSING - add GROQ_API_KEY to .env")
if not key:
    raise SystemExit

llm = LLM(key, model)
print("\n1) Test call")
try:
    print("   Groq replied:", llm.chat("Reply with the word OK.", "Hi"))
except Exception as e:
    print("   FAILED:", friendly_error(e))
    raise SystemExit

print("\n2) Savings Planner")
art = load()
if art is None:
    raise SystemExit("   No trained model yet - run: python train.py")
start = (pd.Timestamp(art.split_date) + pd.Timedelta(days=1)).date()
state = {"analysis": analyze(art, start, art.feature_frame["timestamp"].max().date(), Tariff())}
state["usage"], state["waste"] = _usage_offline(state), _waste_offline(state)
orch = Orchestrator(llm)
plan, used_ai = orch._run("planner", state)
if used_ai:
    print("   Worked. The AI's plan:")
    for a in plan.get("actions", []):
        print(f"   - {a.get('title')}  (${a.get('annual_cost_usd')}/yr)")
else:
    print("   FAILED:", orch.trace[-1]["error"])
