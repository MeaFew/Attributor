"""Streamlit dashboard for Marketing Attribution & Budget Optimization."""

import json

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from attributor.config import IMAGES_DIR, MODEL_OUTPUT_DIR

st.set_page_config(page_title="Marketing Attribution", layout="wide")


@st.cache_data
def load_mmm_results():
    with open(MODEL_OUTPUT_DIR / "mmm_results.json") as f:
        return json.load(f)


@st.cache_data
def load_attribution_results():
    with open(MODEL_OUTPUT_DIR / "attribution_comparison.json") as f:
        return json.load(f)


@st.cache_data
def load_budget_results():
    with open(MODEL_OUTPUT_DIR / "budget_optimization.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["MMM Overview", "Attribution Comparison", "Budget Simulator"])

# ---------------------------------------------------------------------------
# Page 1: MMM Overview
# ---------------------------------------------------------------------------
if page == "MMM Overview":
    st.title("Marketing Mix Modeling Overview")

    try:
        mmm = load_mmm_results()
    except FileNotFoundError:
        st.error("MMM results not found. Run `python -m attributor.mmm_model` first.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    ols = mmm["models"]["ols"]
    ridge = mmm["models"]["ridge"]
    lasso = mmm["models"]["lasso"]

    col1.metric("OLS R2", f"{ols['r2']:.3f}")
    col2.metric("Ridge R2", f"{ridge['r2']:.3f}")
    col3.metric("Lasso R2", f"{lasso['r2']:.3f}")

    st.subheader("Model Diagnostics")
    diag_col1, diag_col2 = st.columns(2)
    diag_col1.write(f"**Durbin-Watson:** {ols['durbin_watson']:.3f}")
    diag_col1.write(f"**AIC:** {ols['aic']:.1f}")
    diag_col2.write(f"**BIC:** {ols['bic']:.1f}")

    # VIF table
    st.subheader("Variance Inflation Factor (VIF)")
    vif_data = ols.get("vif", [])
    if vif_data:
        vif_df = pl.DataFrame(vif_data).sort("vif", descending=True)
        st.dataframe(vif_df.to_pandas(), use_container_width=True)

    # Coefficient comparison chart
    st.subheader("Channel Coefficient Comparison")
    coef_data = []
    for model_name, model_data in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
        for feat, info in model_data["coefficients"].items():
            if "adstock" in feat:
                coef_data.append(
                    {
                        "Channel": feat.replace("_adstock", "")
                        .replace("google_", "G_")
                        .replace("meta_", "M_"),
                        "Model": model_name,
                        "Coefficient": info["coef"],
                    }
                )
    if coef_data:
        coef_df = pl.DataFrame(coef_data)
        fig = px.bar(
            coef_df.to_pandas(),
            x="Channel",
            y="Coefficient",
            color="Model",
            barmode="group",
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Residual diagnostics images
    st.subheader("Residual Diagnostics")
    res_img = IMAGES_DIR / "mmm_residual_diagnostics.png"
    if res_img.exists():
        st.image(str(res_img), use_container_width=True)

# ---------------------------------------------------------------------------
# Page 2: Attribution Comparison
# ---------------------------------------------------------------------------
if page == "Attribution Comparison":
    st.title("Multi-Touch Attribution Model Comparison")

    try:
        attr = load_attribution_results()
    except FileNotFoundError:
        st.error(
            "Attribution results not found. Run `python -m attributor.multi_touch_attribution` first."
        )
        st.stop()

    # Prepare data for stacked bar chart
    model_names = list(attr.keys())
    channels = sorted(set(ch for v in attr.values() for ch in v))

    fig = go.Figure()
    for model in model_names:
        values = [attr[model].get(ch, 0) for ch in channels]
        fig.add_trace(
            go.Bar(
                name=model,
                x=channels,
                y=values,
            )
        )

    fig.update_layout(
        barmode="group",
        title="Attribution Share by Model (%)",
        xaxis_title="Channel",
        yaxis_title="Attributed %",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    st.subheader("Detailed Comparison")
    rows = []
    for ch in channels:
        row = {"Channel": ch}
        for model in model_names:
            row[model] = f"{attr[model].get(ch, 0):.1f}%"
        rows.append(row)
    st.dataframe(pl.DataFrame(rows).to_pandas(), use_container_width=True)

# ---------------------------------------------------------------------------
# Page 3: Budget Simulator
# ---------------------------------------------------------------------------
if page == "Budget Simulator":
    st.title("Budget Allocation Simulator")

    try:
        budget = load_budget_results()
    except FileNotFoundError:
        st.error("Budget results not found. Run `python -m attributor.budget_optimizer` first.")
        st.stop()

    scenario = st.selectbox("Select scenario", list(budget.keys()))
    result = budget[scenario]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Budget", f"${result['total_budget']:,.0f}")
    col2.metric("Current Revenue", f"${result['current_revenue']:,.0f}")
    col3.metric(
        "Optimal Revenue",
        f"${result['optimal_revenue']:,.0f}",
        delta=f"{result['improvement_pct']:.1f}%",
    )

    # Budget allocation comparison
    st.subheader("Budget Allocation")
    channels = result["channels"]
    current = [result["current_spend"][c] for c in channels]
    optimal = [result["optimal_spend"][c] for c in channels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=channels, y=current))
    fig.add_trace(go.Bar(name="Optimal", x=channels, y=optimal))
    fig.update_layout(
        barmode="group",
        title="Current vs Optimal Budget Allocation",
        xaxis_title="Channel",
        yaxis_title="Spend ($)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Interactive budget slider
    st.subheader("Interactive Budget Adjustment")
    st.write("Adjust each channel's budget and see the predicted revenue impact.")

    adjusted = {}
    cols = st.columns(3)
    for i, ch in enumerate(channels):
        with cols[i % 3]:
            adjusted[ch] = st.slider(
                ch.replace("_spend", "").replace("google_", "G_").replace("meta_", "M_"),
                min_value=0,
                max_value=int(result["current_spend"][ch] * 5),
                value=int(result["current_spend"][ch]),
                step=50,
            )

    # Hill saturation prediction consistent with budget_optimizer.optimize_budget
    try:
        import numpy as np

        from attributor.budget_optimizer import hill_response
        from attributor.config import HILL_GAMMA

        mmm = load_mmm_results()
        ridge = mmm["models"]["ridge"]["coefficients"]
        ridge_intercept = mmm["models"]["ridge"].get("intercept", 0.0)

        # Build per-channel arrays matching budget_optimizer's Hill model
        spend_vals = np.array([float(adjusted[ch]) for ch in channels])
        elasticities = np.array(
            [ridge.get(ch.replace("_spend", "_adstock"), {}).get("coef", 0) for ch in channels]
        )
        # tau = current average spend (half-saturation point), same as optimizer
        tau = np.array([float(result["current_spend"][ch]) for ch in channels])

        # Hill saturation formula is shared with budget_optimizer.hill_response
        # (single source of truth — do not re-implement it here).
        hill_rev = hill_response(spend_vals, elasticities, tau, HILL_GAMMA)
        predicted = float(np.sum(hill_rev) + ridge_intercept)

        current_pred = result["current_revenue"]
        st.metric(
            "Predicted Revenue (Hill saturation)",
            f"${predicted:,.0f}",
            delta=f"{(predicted - current_pred) / abs(current_pred) * 100:.1f}%"
            if current_pred != 0
            else "N/A",
        )
    except (FileNotFoundError, KeyError, TypeError) as e:
        st.warning(f"Failed to load MMM results: {e}")
        st.info("Load MMM results for revenue prediction.")
