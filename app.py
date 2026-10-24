"""AI Factory Energy Optimizer: Streamlit app.

Run:  streamlit run app.py
Settings (API key and model) are read from the .env file and never shown in the app.
"""
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:  # read GROQ_API_KEY and GROQ_MODEL from .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.agents import DEFAULT_MODEL, LLM, Orchestrator
from src.data import FEATURE_NAMES, load_data
from src.ml import Tariff, analyze, load, save, train

st.set_page_config(page_title="AI Factory Energy Optimizer", page_icon="⚡", layout="wide")
BLUE, GREY, RED, GREEN = "#3b6fd8", "#9aa0a6", "#d64545", "#2e9e6a"


def setting(name: str, default: str = "") -> str:
    """From .env locally, or from Streamlit secrets when deployed."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


@st.cache_resource(show_spinner="Training the models (first run only)...")
def get_models():
    art = load()
    if art is None:
        raw, source = load_data()
        art = train(raw, source)
        save(art)
    return art


art = get_models()
test_start = (pd.Timestamp(art.split_date) + pd.Timedelta(days=1)).date()
data_end = art.feature_frame["timestamp"].max().date()
llm = LLM(setting("GROQ_API_KEY") or None, setting("GROQ_MODEL") or DEFAULT_MODEL)

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"The models were trained on data up to {art.split_date} and never saw "
               f"{test_start} to {data_end}, so that's the period analysed here.")
    period = st.date_input("Period to analyse", value=(test_start, data_end),
                           min_value=test_start, max_value=data_end)

    st.subheader("Electricity prices")
    peak_price = st.number_input("Peak price ($ per kWh)", 0.01, 1.0, 0.18, 0.01)
    offpeak_price = st.number_input("Off-peak price ($ per kWh)", 0.01, 1.0, 0.09, 0.01)
    peak_hours = st.slider("Peak hours on weekdays", 0, 24, (10, 18))

    with st.expander("Savings assumptions"):
        idle_share = st.slider("Share of 'plant closed' waste that can be switched off", 0.1, 1.0, 0.5, 0.05)
        shift_share = st.slider("Share of heavy peak-hour work that can move", 0.0, 0.5, 0.15, 0.05)

if not isinstance(period, (tuple, list)) or len(period) != 2:
    st.info("Pick a start and an end date in the sidebar.")
    st.stop()

tariff = Tariff(peak_price, offpeak_price, peak_hours[0], peak_hours[1])
result = analyze(art, period[0], period[1], tariff, idle_share, shift_share)
p, s = result["frame"], result["summary"]

# Clear old agent results when the settings change
run_key = json.dumps([str(period), peak_price, offpeak_price, peak_hours, idle_share, shift_share])
if st.session_state.get("run_key") != run_key:
    st.session_state.update(run_key=run_key, state=None)

# ----------------------------------------------------------------- Header
st.title("⚡ AI Factory Energy Optimizer")
st.markdown("Finds where a steel plant wastes electricity, and uses 4 AI agents to explain it "
            "and plan fixes, with the **money and CO₂ saved**.")
st.caption(f"Data: {art.data_source} · "
           + ("🟢 AI agents online" if llm.online else "🟡 Offline: add GROQ_API_KEY to .env for AI agents"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Electricity used", f"{s['total_kwh']:,} kWh")
c1.caption(f"over {s['days']} days")
c2.metric("Electricity bill", f"${s['total_cost_usd']:,}")
c3.metric("CO₂ produced", f"{s['total_co2_t']} t")
c4.metric("Possible savings / year", f"${result['total_annual_savings_usd']:,}")
c4.caption(f"and {result['total_annual_co2_t']} t of CO₂ avoided per year")

tab1, tab2, tab3 = st.tabs(["📊 Usage & waste", "🤖 AI agent team", "🧪 How it works"])

# ------------------------------------------------------- Tab 1: the data
with tab1:
    hourly = p.set_index("timestamp")[["usage_kwh", "forecast_kwh"]].resample("1h").sum().reset_index()
    fig = go.Figure()
    fig.add_scatter(x=hourly["timestamp"], y=hourly["usage_kwh"], name="Actual usage",
                    line=dict(color=BLUE, width=1.5))
    fig.add_scatter(x=hourly["timestamp"], y=hourly["forecast_kwh"], name="Expected usage (forecast)",
                    line=dict(color=GREY, width=1.5, dash="dot"))
    spikes = p[p["anomaly"]].set_index("timestamp").resample("1h")["usage_kwh"].sum()
    spikes = hourly.set_index("timestamp").loc[spikes[spikes > 0].index, "usage_kwh"]
    fig.add_scatter(x=spikes.index, y=spikes.values, mode="markers", name="Unusual usage",
                    marker=dict(color=RED, size=7))
    fig.update_layout(title="Actual vs expected electricity use (red = unusual)", yaxis_title="kWh per hour",
                      height=400, margin=dict(t=50, b=10), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Where the savings are")
        opp = pd.DataFrame(result["opportunities"])
        fig = go.Figure(go.Bar(y=opp["opportunity"], x=opp["annual_cost_usd"], orientation="h",
                               marker_color=GREEN, text=[f"${v:,}" for v in opp["annual_cost_usd"]],
                               textposition="auto"))
        fig.update_layout(xaxis_title="$ per year", height=240, margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.dataframe(opp.rename(columns={"opportunity": "Opportunity", "annual_cost_usd": "$ / year",
                                         "annual_co2_t": "t CO₂ / year", "annual_kwh": "kWh / year",
                                         "how_calculated": "How it's calculated"}), hide_index=True)
    with right:
        st.subheader(f"Biggest unusual events ({len(result['events'])})")
        ev = pd.DataFrame(result["events"][:20])
        if not ev.empty:
            ev = ev.rename(columns={"start": "Start", "end": "End", "actual_kwh": "Used (kWh)",
                                    "expected_kwh": "Expected (kWh)", "extra_kwh": "Extra (kWh)",
                                    "when": "When"})
        st.dataframe(ev, hide_index=True, height=420)

# ----------------------------------------------------- Tab 2: the agents
with tab2:
    st.graphviz_chart("""
        digraph { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
          ML [label="scikit-learn\\nnumbers", fillcolor="#e6f4ec"];
          U [label="1. Usage Analyst"]; W [label="2. Waste Detective"];
          P [label="3. Savings Planner"]; R [label="4. Reviewer\\n+ number check", fillcolor="#fdf0e6"];
          S [label="Summary", shape=note, fillcolor="#ffffff"];
          ML -> U -> W -> P -> R; R -> P [label="  fix it (once)", style=dashed]; R -> S [label="approved"]; }
    """)
    st.caption("1 explains usage · 2 explains the waste · 3 writes the plan · "
               "4 checks every number against the data and can send the plan back once.")

    if st.button("▶ Run the AI agent team", type="primary"):
        with st.status("Agents at work...", expanded=True) as status:
            state = Orchestrator(llm).run(result, on_step=lambda name: st.write(f"**{name}** working..."))
            status.update(label="Done", state="complete")
        st.session_state["state"] = state

    state = st.session_state.get("state")
    if not state:
        st.info("Click **Run the AI agent team** to get the findings and an action plan.")
    else:
        review = state["review"]
        st.success(f"**Reviewer:** {review.get('verdict', '')}. {review.get('comment', '')}")
        if review.get("rejected_because"):
            with st.expander("Why the AI plan was rejected"):
                for reason in review["rejected_because"]:
                    st.markdown(f"- {reason}")
        failed = [t for t in state["trace"] if t["error"]]
        for t in failed:
            st.warning(f"**{t['agent']}** used its template instead of AI: {t['error']}")

        st.subheader("Action plan")
        plan = pd.DataFrame(state["plan"].get("actions", []))
        if not plan.empty:
            cols = [c for c in ["title", "owner", "timeline", "annual_cost_usd", "annual_co2_t",
                                "priority", "why"] if c in plan]
            st.dataframe(plan[cols].rename(columns={
                "title": "Action", "owner": "Owner", "timeline": "Timeline", "annual_cost_usd": "$ / year",
                "annual_co2_t": "t CO₂ / year", "priority": "Priority", "why": "Why"}), hide_index=True)

        st.subheader("Summary for the plant manager")
        st.markdown(state["summary"])
        st.download_button("Download summary", state["summary"], "energy_summary.md", "text/markdown")

        st.subheader("What each agent did")
        for t in state["trace"]:
            with st.expander(f"{t['agent']} · {t['job']} · {t['mode']} · {t['seconds']}s"):
                if isinstance(t["output"], dict):
                    st.json(t["output"])
                else:
                    st.markdown(t["output"])
                st.caption("What the agent was given:")
                st.code(t["input"][:3000], language="text")

# Tab 3: the method
with tab3:
    st.markdown(f"""
The problem: A factory pays for electricity but can't see where it's wasted.
This app looks at the plant's meter, which records usage every 15 minutes, and finds three kinds of waste.

Step 1: Predict normal usage (scikit-learn, Random Forest).
The model learns what usage should be at any moment from patterns like last week's and yesterday's
usage at the same time. It was trained on Jan–Oct 2018 and tested on Nov–Dec, which it never saw.

Step 2: Spot unusual usage (scikit-learn, Isolation Forest).
This model flags readings that are unusual for that time of day, such as a big spike on a Sunday
afternoon. Extra energy = actual usage minus the forecast.

Step 3: Calculate the savings (pandas).
- Plant closed: usage at night or on weekends above the normal "everything off" level.
- Unusual spikes: the extra energy in the flagged readings.
- Cheaper hours: moving some heavy work from peak-price hours to off-peak hours (saves money, not energy).

Each reading counts toward only one of these, so nothing is counted twice.
Yearly figures = savings in the period (365 ÷ days in the period).

Step 4: The 4 AI agents (Groq LLM): explain the numbers and write the plan.
The Reviewer and a code check make sure the plan never claims more than the data supports.
""")
    st.subheader("How accurate is the forecast?")
    rows = [{"Method": k, "Typical error (kWh per 15 min)": round(v["MAE"], 2), "R²": round(v["R2"], 3)}
            for k, v in art.metrics.items() if isinstance(v, dict)]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("R² = share of the ups and downs the method explains (1.0 = perfect). "
               "The baselines simply copy yesterday's or last week's value.")
    imp = pd.Series({FEATURE_NAMES.get(k, k): v for k, v in art.feature_importance.items()}).sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLUE))
    fig.update_layout(title="What the forecast relies on most", height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig)
    st.markdown("""
Glossary
- kWh (kilowatt-hour):** the unit on an electricity bill. Running a 1,000-watt heater for an hour uses 1 kWh.
- Peak / off-peak price:** electricity costs more at busy times of day. The prices here are illustrative.
- t CO₂:tonnes of carbon dioxide. Calculated from the CO₂ figures in the dataset (about 0.42 kg per kWh).
""")
