"""
Interactive Streamlit Frontend Dashboard:
Next-Day Rainfall Prediction & Explainable AI (XAI) System for Ratmalana, Sri Lanka.
Integrates the Explainable Boosting Machine (EBM - new untaught algorithm) and the
top 5 lecture-taught ML algorithms with real-time local & global explanations
(EBM Glass-Box Waterfall & Shape Functions, LIME, and SHAP).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model_selection_benchmark import main as run_benchmark_pipeline  # noqa: E402

BUNDLE_PATH = BASE_DIR / "artifacts" / "models_bundle.joblib"
PLOTS_DIR = BASE_DIR / "artifacts" / "plots"
DISCRETE_FEATURES = {"stations_raining", "monsoon_SW", "monsoon_IM1", "monsoon_IM2"}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Keep a slider default inside the slider range."""
    return float(min(max(float(value), lo), hi))


st.set_page_config(
    page_title="Sri Lanka Next-Day Rainfall XAI Forecaster",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_bundle():
    if not BUNDLE_PATH.exists():
        run_benchmark_pipeline()
    return joblib.load(BUNDLE_PATH)


@st.cache_resource
def get_lime_explainer(_X_trval_values: np.ndarray, feature_names: list[str]) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        training_data=_X_trval_values,
        feature_names=feature_names,
        class_names=["Dry (<1mm)", "Rain (>=1mm)"],
        mode="classification",
        random_state=42,
    )


bundle = load_bundle()
models = bundle["models"]
feats = bundle["features"]
feature_labels = bundle["feature_labels"]
feature_stats = bundle["feature_stats"]
val_thresholds = bundle["val_thresholds"]
results_df = bundle["results_df"]
X_trval = bundle["X_trval"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
test_target_mm = bundle["test_target_mm"]
full_df = bundle["full_df"]
xai_data = bundle["xai"]

st.title("🌧️ Sri Lanka Next-Day Rainfall Prediction & Explainable AI (XAI) Dashboard")
st.markdown(
    "**Target Station:** Ratmalana Meteorological Station (Western Province, Sri Lanka) | "
    "**Task:** Predict Next-Day Wet Day ($\\ge 1.0\\text{ mm}$ Rainfall) | "
    "**Featured New Model:** **Explainable Boosting Machine (EBM / $\\text{GA}^2\\text{M}$)** vs. **Top 5 Lecture Algorithms** "
    "| *Decision-support tool only — always follow official Department of Meteorology forecasts.*"
)

# Sidebar Controls for Live Prediction Input
st.sidebar.header("🎛️ Weather Input Controls")
input_mode = st.sidebar.radio(
    "Select Input Mode:",
    ["📅 Pick a 2018 Test Date", "🎚️ Custom Weather Scenario Sliders"],
    index=0,
)

if input_mode == "📅 Pick a 2018 Test Date":
    date_options = [d.strftime("%Y-%m-%d") for d in X_test.index]
    default_date = xai_data.get("sample_wet_date", date_options[120])
    default_idx = date_options.index(default_date) if default_date in date_options else 0
    selected_date = st.sidebar.selectbox("Select 2018 Date (Day t):", date_options, index=default_idx)
    row_idx = date_options.index(selected_date)
    input_series = X_test.iloc[row_idx].copy()
    default_doy = int(X_test.index[row_idx].dayofyear)
    actual_next_rain_mm = float(test_target_mm[row_idx])
    actual_next_label = int(y_test[row_idx])
    st.sidebar.info(
        f"**Selected Date:** `{selected_date}`\n\n"
        f"**Actual Next-Day Rain:** `{actual_next_rain_mm:.1f} mm` "
        f"({'🌧️ Wet Day (>=1mm)' if actual_next_label == 1 else '☀️ Dry Day (<1mm)'})"
    )
    allow_override = st.sidebar.checkbox("Override specific features (What-If Counterfactual Analysis)", value=False)
else:
    selected_date = "Custom Scenario"
    row_idx = 0
    actual_next_rain_mm = None
    actual_next_label = None
    input_series = pd.Series({c: feature_stats[c]["median"] for c in feats})
    input_series["rat_rain_lag1"] = 0.0
    input_series["deh_rain_lag1"] = 0.0
    input_series["hom_rain_lag1"] = 0.0
    input_series["rat_rhmax_change1d"] = 0.0
    input_series["rat_tmax_change1d"] = 0.0
    default_doy = 150
    allow_override = True

if allow_override:
    base_rhmax_yesterday = float(input_series["rat_rhmax"] - input_series["rat_rhmax_change1d"])
    base_tmax_yesterday = float(input_series["rat_tmax"] - input_series["rat_tmax_change1d"])
    base_deh_before_today = float(max(0.0, input_series["deh_rain_3d"] - input_series["deh_rain"]))
    st.sidebar.subheader("🌦️ Primary Meteorological Inputs")
    rat_rain = st.sidebar.slider(
        "Ratmalana Rain Today (mm)", 0.0, 260.0, _clamp(input_series["rat_rain"], 0.0, 260.0), 0.5
    )
    deh_rain = st.sidebar.slider(
        "Dehiwala Rain Today (mm)", 0.0, 260.0, _clamp(input_series["deh_rain"], 0.0, 260.0), 0.5
    )
    hom_rain = st.sidebar.slider(
        "Homagama Rain Today (mm)", 0.0, 260.0, _clamp(input_series["hom_rain"], 0.0, 260.0), 0.5
    )
    rat_rhmax = st.sidebar.slider(
        "Ratmalana Max Humidity (%)", 40.0, 100.0, _clamp(input_series["rat_rhmax"], 40.0, 100.0), 0.5
    )
    col_rhmax = st.sidebar.slider(
        "Colombo Max Humidity (%)", 40.0, 100.0, _clamp(input_series["col_rhmax"], 40.0, 100.0), 0.5
    )
    rat_tmax = st.sidebar.slider(
        "Ratmalana Max Temp (°C)", 18.0, 38.0, _clamp(input_series["rat_tmax"], 18.0, 38.0), 0.1
    )
    col_tmax = st.sidebar.slider(
        "Colombo Max Temp (°C)", 18.0, 38.0, _clamp(input_series["col_tmax"], 18.0, 38.0), 0.1
    )
    rat_rain_lag1 = st.sidebar.slider(
        "Ratmalana Rain Yesterday (mm)", 0.0, 260.0, _clamp(input_series["rat_rain_lag1"], 0.0, 260.0), 0.5
    )
    rat_rain_3d = st.sidebar.slider(
        "Ratmalana 3-Day Cumulative Rain (mm)",
        float(rat_rain),
        400.0,
        _clamp(input_series["rat_rain_3d"], float(rat_rain), 400.0),
        1.0,
    )
    rat_rain_7d = st.sidebar.slider(
        "Ratmalana 7-Day Cumulative Rain (mm)",
        float(rat_rain_3d),
        600.0,
        _clamp(input_series["rat_rain_7d"], float(rat_rain_3d), 600.0),
        1.0,
    )
    deh_rain_lag1 = st.sidebar.slider(
        "Dehiwala Rain Yesterday (mm)", 0.0, 260.0, _clamp(input_series["deh_rain_lag1"], 0.0, 260.0), 0.5
    )
    hom_rain_lag1 = st.sidebar.slider(
        "Homagama Rain Yesterday (mm)", 0.0, 260.0, _clamp(input_series["hom_rain_lag1"], 0.0, 260.0), 0.5
    )
    day_of_year = st.sidebar.slider("Day of Year (1 - 365)", 1, 366, default_doy)

    # Recompute derived spatial & seasonal features automatically
    input_series["rat_rain"] = rat_rain
    input_series["deh_rain"] = deh_rain
    input_series["hom_rain"] = hom_rain
    input_series["regional_rain_mean"] = np.mean([rat_rain, deh_rain, hom_rain])
    input_series["stations_raining"] = float(sum(x >= 1.0 for x in [rat_rain, deh_rain, hom_rain]))
    input_series["rat_rhmax"] = rat_rhmax
    input_series["col_rhmax"] = col_rhmax
    input_series["rh_diff_rat_col"] = rat_rhmax - col_rhmax
    input_series["rat_tmax"] = rat_tmax
    input_series["col_tmax"] = col_tmax
    input_series["tmax_diff_rat_col"] = rat_tmax - col_tmax
    input_series["rat_rain_lag1"] = rat_rain_lag1
    input_series["rat_rain_3d"] = rat_rain_3d
    input_series["rat_rain_7d"] = rat_rain_7d
    input_series["deh_rain_lag1"] = deh_rain_lag1
    input_series["hom_rain_lag1"] = hom_rain_lag1
    deh_t2 = max(0.0, base_deh_before_today - float(X_test.iloc[row_idx]["deh_rain_lag1"])) if input_mode == "📅 Pick a 2018 Test Date" else 0.0
    input_series["deh_rain_3d"] = deh_rain + deh_rain_lag1 + deh_t2
    input_series["rat_rhmax_change1d"] = rat_rhmax - base_rhmax_yesterday
    input_series["rat_tmax_change1d"] = rat_tmax - base_tmax_yesterday
    input_series["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    input_series["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    approx_month = int((pd.Timestamp("2018-01-01") + pd.Timedelta(days=min(day_of_year, 365) - 1)).month)
    input_series["monsoon_SW"] = float(approx_month in [5, 6, 7, 8, 9])
    input_series["monsoon_IM1"] = float(approx_month in [3, 4])
    input_series["monsoon_IM2"] = float(approx_month in [10, 11])

input_df = pd.DataFrame([input_series[feats]], columns=feats)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Forecast & Local XAI (EBM + LIME + SHAP)",
    "📊 Model Benchmark (New EBM vs. Top 5 Lecture Models)",
    "🧠 Global Explainability (EBM Shape Curves & SHAP)",
    "🗺️ Sri Lanka Western Province Dataset Explorer",
])

with tab1:
    ebm_model = models["[NEW] Explainable Boosting Machine"]
    ebm_prob = float(ebm_model.predict_proba(input_df)[0, 1])
    ebm_thresh = val_thresholds["[NEW] Explainable Boosting Machine"]
    ebm_pred = int(ebm_prob >= ebm_thresh)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(
            label="[NEW] EBM Next-Day Rain Probability",
            value=f"{ebm_prob:.1%}",
            delta=f"Threshold: {ebm_thresh:.2f} ({'🌧️ RAIN' if ebm_pred == 1 else '☀️ DRY'})",
        )
    with col_b:
        rf_prob = float(models["[L6] Random Forest"].predict_proba(input_df)[0, 1])
        rf_thresh = val_thresholds["[L6] Random Forest"]
        st.metric(
            label="[Best Lecture Model] Random Forest Rain Probability",
            value=f"{rf_prob:.1%}",
            delta=f"Threshold: {rf_thresh:.2f} ({'🌧️ RAIN' if rf_prob >= rf_thresh else '☀️ DRY'})",
        )
    with col_c:
        if actual_next_label is not None:
            st.metric(
                label=f"Observed Outcome ({selected_date} + 1d)",
                value="🌧️ Wet Day (>=1mm)" if actual_next_label == 1 else "☀️ Dry Day (<1mm)",
                delta=f"Actual Rainfall: {actual_next_rain_mm:.1f} mm",
            )
        else:
            st.metric(
                label="Regional Stations Raining Today",
                value=f"{int(input_series['stations_raining'])} / 3 Stations",
                delta=f"Mean Regional Rain: {input_series['regional_rain_mean']:.1f} mm",
            )

    st.subheader("📋 Multi-Model Consensus for Current Input")
    pred_rows = []
    for m_name, mdl in models.items():
        p_val = float(mdl.predict_proba(input_df)[0, 1])
        t_val = val_thresholds[m_name]
        pred_rows.append({
            "Model": m_name,
            "Category": "🌟 New Algorithm (Untaught)" if m_name.startswith("[NEW") else "📚 Lecture Algorithm",
            "Predicted Rain Prob": f"{p_val:.1%}",
            "Val-Tuned Threshold": f"{t_val:.3f}",
            "Decision": "🌧️ Rain (>=1.0 mm)" if p_val >= t_val else "☀️ Dry (<1.0 mm)",
        })
    st.dataframe(pd.DataFrame(pred_rows), width="stretch", hide_index=True)

    st.subheader("🔍 Local Explainability: Why Did EBM Make This Prediction?")
    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.markdown("#### 1. EBM Exact Glass-Box Local Additive Contributions")
        ebm_local = ebm_model.explain_local(input_df, [actual_next_label if actual_next_label is not None else ebm_pred])
        loc_data = ebm_local.data(0)
        l_names = loc_data["names"]
        l_scores = np.array(loc_data["scores"], dtype=float)
        l_values = loc_data["values"]
        top_loc_idx = np.argsort(np.abs(l_scores))[-10:]

        fig_ebm_loc, ax_loc = plt.subplots(figsize=(7.5, 4.8))
        bar_labels = []
        for i in top_loc_idx:
            nm = l_names[i]
            val = l_values[i]
            pretty = feature_labels.get(nm, nm)
            if isinstance(val, (int, float, np.floating)):
                bar_labels.append(f"{pretty} = {val:.2f}")
            else:
                bar_labels.append(f"{pretty} ({val})")
        colors_loc = ["#D95F02" if s > 0 else "#1F78B4" for s in l_scores[top_loc_idx]]
        ax_loc.barh(range(len(top_loc_idx)), l_scores[top_loc_idx], color=colors_loc)
        ax_loc.axvline(0.0, color="black", lw=0.9, ls="--")
        ax_loc.set_yticks(range(len(top_loc_idx)))
        ax_loc.set_yticklabels(bar_labels, fontsize=8.5)
        ax_loc.set_xlabel("Exact Additive Log-Odds Contribution (Orange = +Rain, Blue = -Rain)")
        ax_loc.set_title(f"EBM Local Breakdown (Intercept = {loc_data['extra']['scores'][0]:.2f})")
        plt.tight_layout()
        st.pyplot(fig_ebm_loc)
        plt.close(fig_ebm_loc)

    with xai_col2:
        st.markdown("#### 2. LIME Local Linear Surrogate Explanation")
        lime_explainer = get_lime_explainer(
            X_trval.values,
            [feature_labels.get(c, c) for c in feats],
        )
        lime_exp = lime_explainer.explain_instance(
            input_df.iloc[0].values,
            ebm_model.predict_proba,
            num_features=10,
        )
        lime_list = lime_exp.as_list()
        lime_rules = [x[0] for x in lime_list][::-1]
        lime_weights = [x[1] for x in lime_list][::-1]

        fig_lime, ax_lime = plt.subplots(figsize=(7.5, 4.8))
        c_lime = ["#2CA02C" if w > 0 else "#D62728" for w in lime_weights]
        ax_lime.barh(range(len(lime_rules)), lime_weights, color=c_lime)
        ax_lime.axvline(0.0, color="black", lw=0.9, ls="--")
        ax_lime.set_yticks(range(len(lime_rules)))
        ax_lime.set_yticklabels(lime_rules, fontsize=8.5)
        ax_lime.set_xlabel("LIME Surrogate Weight on P(Rain)")
        ax_lime.set_title("LIME Local Perturbation Surrogate Weights")
        plt.tight_layout()
        st.pyplot(fig_lime)
        plt.close(fig_lime)

with tab2:
    st.subheader("🏆 Benchmark Comparison: Explainable Boosting Machine (EBM) vs. Lecture Models")
    st.markdown(
        "Hyperparameters chosen by **rolling-origin time-series CV** (validate on 2014, 2015, 2016, 2017, each time "
        "training only on earlier years; `CV_ROC-AUC` = mean) $\\rightarrow$ decision threshold tuned on **Validation (2017)** "
        "(max F1) $\\rightarrow$ refit on 2012–2017 $\\rightarrow$ **Unseen Test (2018)** ($N = 364$ days), evaluated once."
    )
    _r = results_df.set_index("model")
    _ebm = "[NEW] Explainable Boosting Machine"
    _cv_best = _r["CV_ROC-AUC"].idxmax()
    _te_best = _r["ROC-AUC"].idxmax()
    _cv_beaten = [m.split("] ")[1] for m in _r.index if m.startswith("[L") and _r.loc[m, "CV_ROC-AUC"] < _r.loc[_ebm, "CV_ROC-AUC"]]
    st.info(
        f"**Reading the table:** across the four rolling-origin CV years EBM scores {_r.loc[_ebm, 'CV_ROC-AUC']:.3f} ROC-AUC, "
        f"{_r.loc[_cv_best, 'CV_ROC-AUC'] - _r.loc[_ebm, 'CV_ROC-AUC']:.3f} below the best ({_cv_best.split('] ')[1]}) and ahead of "
        f"{', '.join(_cv_beaten)}. On the single 2018 test year it scores {_r.loc[_ebm, 'ROC-AUC']:.3f} vs "
        f"{_r.loc[_te_best, 'ROC-AUC']:.3f} for {_te_best.split('] ')[1]} — gaps of this size on 364 days are close to noise, "
        "and are the price of a fully transparent additive model. Recall/precision depend on each model's "
        "validation-tuned threshold (lower threshold = more wet days caught, more false alarms)."
    )
    display_df = results_df.copy()
    st.dataframe(
        display_df.style.format({
            "CV_ROC-AUC": "{:.4f}",
            "Val_ROC-AUC": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
            "F1": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "Accuracy": "{:.4f}",
            "Brier": "{:.4f}",
            "Opt_Threshold": "{:.3f}",
        }),
        width="stretch",
        hide_index=True,
    )

    if (PLOTS_DIR / "model_comparison_roc_pr.png").exists():
        st.image(
            str(PLOTS_DIR / "model_comparison_roc_pr.png"),
            caption="ROC Curves, Precision-Recall Curves, and Probability Calibration on the 2018 Test Set",
            width="stretch",
        )
    if (PLOTS_DIR / "confusion_matrices.png").exists():
        st.image(
            str(PLOTS_DIR / "confusion_matrices.png"),
            caption="Confusion Matrices on the 2018 Test Set",
            width="stretch",
        )

with tab3:
    st.subheader("🧠 Global Model Interpretability (EBM Glass-Box Shape Functions & SHAP)")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if (PLOTS_DIR / "ebm_global_importance.png").exists():
            st.image(
                str(PLOTS_DIR / "ebm_global_importance.png"),
                caption="EBM Exact Global Feature & Pairwise Interaction Importance",
                width="stretch",
            )
    with col_g2:
        if (PLOTS_DIR / "shap_summary_ebm.png").exists():
            st.image(
                str(PLOTS_DIR / "shap_summary_ebm.png"),
                caption="SHAP (Permutation) on the EBM — impact on P(Rain), 2018 Test Set",
                width="stretch",
            )
        if (PLOTS_DIR / "shap_summary_beeswarm.png").exists():
            st.image(
                str(PLOTS_DIR / "shap_summary_beeswarm.png"),
                caption="Cross-model check: TreeSHAP on CatBoost (2018 Test Set)",
                width="stretch",
            )

    st.markdown("#### 📈 Interactive EBM Shape Function Explorer $f_i(x_i)$")
    shapes_dict = xai_data.get("ebm_shapes", {})
    if shapes_dict:
        selected_feat = st.selectbox(
            "Select a Meteorological Feature to Inspect Its Learned Non-Linear Response Curve:",
            options=list(shapes_dict.keys()),
            format_func=lambda k: f"{feature_labels.get(k, k)} ({k})",
        )
        sh = shapes_dict[selected_feat]
        fig_sh, ax_sh = plt.subplots(figsize=(10, 4.2))
        if selected_feat in DISCRETE_FEATURES:
            xs = np.array(sh["x"])
            ys = np.array(sh["scores"])
            lo = np.array(sh["lower"])
            hi = np.array(sh["upper"])
            cats = [int(round(v)) for v in xs]
            cur = int(round(float(input_series[selected_feat])))
            ax_sh.bar(
                range(len(cats)), ys, yerr=[ys - lo, hi - ys], capsize=6,
                color=["#1F78B4" if c == cur else "#D95F02" for c in cats],
                label="EBM contribution (blue = current input)",
            )
            ax_sh.set_xticks(range(len(cats)))
            ax_sh.set_xticklabels(["No (0)", "Yes (1)"] if selected_feat.startswith("monsoon") else [str(c) for c in cats])
            ax_sh.axhline(0.0, color="black", ls="--", lw=1.0)
        else:
            ax_sh.step(sh["x"], sh["scores"], where="mid", color="#D95F02", lw=2.5, label="EBM Shape Function f(x)")
            ax_sh.fill_between(
                sh["x"], sh["lower"], sh["upper"],
                step="mid", color="#D95F02", alpha=0.22, label="±1 SD Uncertainty Band across Outer Bags",
            )
            ax_sh.axhline(0.0, color="black", ls="--", lw=1.0)
            ax_sh.axvline(float(input_series[selected_feat]), color="#1F78B4", ls="-.", lw=1.8, label=f"Current Input ({input_series[selected_feat]:.2f})")
        ax_sh.set_title(f"EBM Learned Shape Function: {feature_labels.get(selected_feat, selected_feat)}", fontweight="bold")
        ax_sh.set_xlabel(feature_labels.get(selected_feat, selected_feat))
        ax_sh.set_ylabel("Log-Odds Contribution to Next-Day Rain")
        ax_sh.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig_sh)
        plt.close(fig_sh)

    if (PLOTS_DIR / "ebm_shape_functions.png").exists():
        st.image(
            str(PLOTS_DIR / "ebm_shape_functions.png"),
            caption="Top 6 EBM Non-Linear Shape Functions f_i(x_i)",
            width="stretch",
        )

with tab4:
    st.subheader("🗺️ Western Province Meteorological Dataset Explorer (2012–2018)")
    st.markdown(
        "Compiled from Sri Lanka Department of Meteorology daily observations across **Ratmalana** (target station), "
        "**Colombo**, **Dehiwala**, and **Homagama** ($2,556$ valid daily records from 2012–2018)."
    )
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Total Daily Records", f"{len(full_df):,}")
    m_col2.metric("Engineered Features", f"{len(feats)}")
    m_col3.metric("Wet Days (>= 1.0 mm)", f"{int(full_df['target'].sum()):,} ({full_df['target'].mean():.1%})")
    m_col4.metric("Dry Days (< 1.0 mm)", f"{int((full_df['target'] == 0).sum()):,} ({1 - full_df['target'].mean():.1%})")

    st.markdown("#### Daily Rainfall Time Series Across Stations (2012–2018)")
    st.line_chart(full_df[["rat_rain", "deh_rain", "hom_rain"]].tail(365))

    st.markdown("#### Sample Processed Feature Matrix (Latest 15 Days)")
    st.dataframe(full_df[feats + ["target"]].tail(15).round(2), width="stretch")
