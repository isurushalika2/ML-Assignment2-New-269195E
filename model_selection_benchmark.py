"""
Benchmark, Training & Explainable AI (XAI) Pipeline:
Next-day rainfall (>= 1.0 mm) prediction at Ratmalana, Western Province, Sri Lanka (2012-2018).
Compares standard supervised machine learning algorithms against the selected
Explainable Boosting Machine (EBM / GA2M) and secondary candidate models
(Extra Trees, Gaussian Process Classifier).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "data"
OUT_CSV = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE_DIR / "model_selection_results.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TUNING_CSV = BASE_DIR / "hyperparameter_tuning_results.csv"
INVALID_DATE_LOG: dict[str, int] = {}

# Binary / count features: their EBM shape functions are drawn as bar charts, not step curves
DISCRETE_FEATURES = {"stations_raining", "monsoon_SW", "monsoon_IM1", "monsoon_IM2"}


def load_station_series(data_dir: Path, fname: str, station: str | None = None) -> pd.DataFrame:
    """Load a meteorological CSV file, deduplicate, and pivot elements by date."""
    df_raw = pd.read_csv(data_dir / fname, encoding="utf-8-sig")
    val_col = df_raw.columns[-1]
    df_raw["date"] = pd.to_datetime(df_raw[["year", "month", "day"]], errors="coerce")
    n_invalid = int(df_raw["date"].isna().sum())
    if n_invalid:
        # e.g. 2012-02-30, 2012-04-31: calendar-impossible dates present in the raw Met Dept exports
        INVALID_DATE_LOG[fname] = n_invalid  # files read twice are not double-counted
    df_raw = df_raw.dropna(subset=["date"])
    if station is not None:
        df_raw = df_raw[df_raw["station_name"] == station]
    df_raw[val_col] = pd.to_numeric(df_raw[val_col], errors="coerce")
    df_raw = df_raw.drop_duplicates(subset=["date", "station_name", "element_name"])
    pivoted = df_raw.pivot_table(
        index="date", columns="element_name", values=val_col, aggfunc="mean"
    )
    return pivoted


def build_dataset(data_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    Build the clean, leak-free 2012-2018 daily meteorological dataset for Ratmalana.
    Fixes:
      1. Recovers 2015 Homagama rainfall stored inside dehiwala_rf_2012-2018.csv.
      2. Deduplicates 2018 Colombo RHMAX entries.
      3. Excludes col_TMPMIN (0 rows in 2018 test set and 2012-2014) and col_RHMIN
         (0 rows in 2012-2014) to prevent 100% test-split imputation distortion.
      4. Replaces 0.0 C sensor errors in Ratmalana TMPMAX with NaN.
    """
    rat_rf = load_station_series(data_dir, "ratmalana_rf_2012-2018.csv")["PRECIP"].rename("rat_rain")
    rat_rh = load_station_series(data_dir, "ratmalana_rh_2012-2018.csv")["RHMAX"].rename("rat_rhmax")
    rat_t = load_station_series(data_dir, "ratmalana_t_2012-2018.csv")["TMPMAX"].rename("rat_tmax")
    rat_t[rat_t < 15.0] = np.nan  # Remove 0.0 C sensor dropouts

    col_rh = load_station_series(data_dir, "colombo_rh_2012-2018.csv")["RHMAX"].rename("col_rhmax")
    col_t = load_station_series(data_dir, "colombo_t_2012-2018.csv")["TMPMAX"].rename("col_tmax")

    deh = load_station_series(data_dir, "dehiwala_rf_2012-2018.csv", "DEHIWALA ZOO")["PRECIP"].rename("deh_rain")

    # Recover 2015 Homagama rainfall from dehiwala_rf_2012-2018.csv and merge with homagama_rf_2012-2018.csv
    hom_main = load_station_series(data_dir, "homagama_rf_2012-2018.csv", "HOMAGAMA")["PRECIP"]
    hom_2015 = load_station_series(data_dir, "dehiwala_rf_2012-2018.csv", "HOMAGAMA")["PRECIP"]
    hom = hom_main.combine_first(hom_2015).rename("hom_rain")

    idx = pd.date_range("2012-01-01", "2018-12-31", name="date")
    df = pd.concat([rat_rf, rat_rh, rat_t, col_rh, col_t, deh, hom], axis=1).reindex(idx)

    # Spatial & cross-station features (using only day-t observations)
    df["regional_rain_mean"] = df[["rat_rain", "deh_rain", "hom_rain"]].mean(axis=1)
    df["stations_raining"] = (df[["rat_rain", "deh_rain", "hom_rain"]] >= 1.0).sum(axis=1).astype(float)
    df["rh_diff_rat_col"] = df["rat_rhmax"] - df["col_rhmax"]
    df["tmax_diff_rat_col"] = df["rat_tmax"] - df["col_tmax"]

    # Temporal lag & rolling persistence features (strictly up to day t)
    df["rat_rain_lag1"] = df["rat_rain"].shift(1)
    df["deh_rain_lag1"] = df["deh_rain"].shift(1)
    df["hom_rain_lag1"] = df["hom_rain"].shift(1)
    df["rat_rain_3d"] = df["rat_rain"].rolling(3, min_periods=1).sum()
    df["rat_rain_7d"] = df["rat_rain"].rolling(7, min_periods=1).sum()
    df["deh_rain_3d"] = df["deh_rain"].rolling(3, min_periods=1).sum()
    df["rat_rhmax_change1d"] = df["rat_rhmax"].diff(1)
    df["rat_tmax_change1d"] = df["rat_tmax"].diff(1)

    # Cyclical seasonality & Sri Lanka monsoon regime indicators
    doy = df.index.dayofyear
    df["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)

    m = df.index.month
    df["monsoon_SW"] = m.isin([5, 6, 7, 8, 9]).astype(float)      # Southwest Monsoon (May-Sep)
    df["monsoon_IM1"] = m.isin([3, 4]).astype(float)              # First Inter-Monsoon (Mar-Apr)
    df["monsoon_IM2"] = m.isin([10, 11]).astype(float)            # Second Inter-Monsoon (Oct-Nov)
    # Northeast Monsoon (Dec-Feb) is the reference category to avoid collinearity

    # Target: Next-day rainfall >= 1.0 mm at Ratmalana (WMO wet-day definition)
    next_rain = df["rat_rain"].shift(-1)
    df["target_mm"] = next_rain
    df["target"] = (next_rain >= 1.0).astype(float)
    df.loc[next_rain.isna(), "target"] = np.nan
    df = df.dropna(subset=["target", "rat_rain"]).copy()

    feature_labels = {
        "rat_rain": "Ratmalana Rainfall Today (mm)",
        "deh_rain": "Dehiwala Rainfall Today (mm)",
        "hom_rain": "Homagama Rainfall Today (mm)",
        "regional_rain_mean": "Regional Mean Rainfall Today (mm)",
        "stations_raining": "Active Rain Stations Today (0-3)",
        "rat_rhmax": "Ratmalana Max Relative Humidity (%)",
        "col_rhmax": "Colombo Max Relative Humidity (%)",
        "rh_diff_rat_col": "Humidity Gradient: Ratmalana - Colombo (%)",
        "rat_tmax": "Ratmalana Max Temperature (°C)",
        "col_tmax": "Colombo Max Temperature (°C)",
        "tmax_diff_rat_col": "Temp Gradient: Ratmalana - Colombo (°C)",
        "rat_rain_lag1": "Ratmalana Rainfall Yesterday (mm)",
        "deh_rain_lag1": "Dehiwala Rainfall Yesterday (mm)",
        "hom_rain_lag1": "Homagama Rainfall Yesterday (mm)",
        "rat_rain_3d": "Ratmalana 3-Day Cumulative Rain (mm)",
        "rat_rain_7d": "Ratmalana 7-Day Cumulative Rain (mm)",
        "deh_rain_3d": "Dehiwala 3-Day Cumulative Rain (mm)",
        "rat_rhmax_change1d": "Ratmalana 24h Humidity Change (%)",
        "rat_tmax_change1d": "Ratmalana 24h Temp Change (°C)",
        "doy_sin": "Seasonality Sine (Day of Year)",
        "doy_cos": "Seasonality Cosine (Day of Year)",
        "monsoon_SW": "Southwest Monsoon (May-Sep)",
        "monsoon_IM1": "First Inter-Monsoon (Mar-Apr)",
        "monsoon_IM2": "Second Inter-Monsoon (Oct-Nov)",
    }
    feats = list(feature_labels.keys())
    return df, feats, feature_labels


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find probability threshold on validation set that maximizes F1-score."""
    thresholds = np.linspace(0.25, 0.70, 91)
    best_t, best_f1 = 0.50, -1.0
    for t in thresholds:
        score = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return round(best_t, 3)


def build_model(name: str, params: dict):
    """Build one estimator from its model family name and a hyperparameter combination."""
    p = dict(params)
    if name == "Explainable Boosting Machine (EBM)":
        return ExplainableBoostingClassifier(
            max_bins=128, max_rounds=4000, early_stopping_rounds=100,
            outer_bags=8, inner_bags=0, random_state=42, n_jobs=-1, **p,
        )
    if name == "CatBoost":
        return CatBoostClassifier(iterations=500, verbose=0, random_seed=42, allow_writing_files=False, **p)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1, **p)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=400, subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=1, **p,
        )
    if name == "SVM (RBF)":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42, **p)),
        ])
    if name == "Decision Tree":
        return DecisionTreeClassifier(criterion="entropy", random_state=42, **p)
    if name == "Logistic Regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=42, **p)),
        ])
    if name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, **p)
    if name == "Gaussian Process Classifier":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianProcessClassifier(kernel=1.0 * RBF(3.0), random_state=42, **p)),
        ])
    raise ValueError(name)


def get_param_grids() -> dict[str, list[dict]]:
    """
    Small, theory-driven hyperparameter grids for each model family.
    Every combination is scored by rolling-origin time-series CV (validate on 2014, 2015, 2016, 2017,
    each time training only on earlier years) and the best mean ROC-AUC wins. One validation year alone
    (365 days) is too noisy to separate configurations. The winner's decision threshold is tuned on 2017,
    then it is refitted on Train+Val (2012-2017) and evaluated once on Test (2018).
    """
    def grid(**axes):
        keys = list(axes)
        combos = [[]]
        for k in keys:
            combos = [c + [v] for c in combos for v in axes[k]]
        return [dict(zip(keys, c)) for c in combos]

    return {
        # EBM: boosting step size, pairwise interactions (GA2M vs pure GAM), and bin smoothness
        "Explainable Boosting Machine (EBM)": grid(
            learning_rate=[0.015, 0.04], interactions=[0, 10], min_samples_leaf=[10, 25],
        ),
        # CatBoost: tree depth, learning rate, L2 leaf regularisation
        "CatBoost": grid(depth=[4, 6], learning_rate=[0.03, 0.06], l2_leaf_reg=[3.0, 8.0]),
        # Random Forest: depth and leaf size control variance of each tree
        "Random Forest": grid(max_depth=[6, 10, None], min_samples_leaf=[3, 10]),
        # XGBoost: depth, learning rate, L2 penalty
        "XGBoost": grid(max_depth=[3, 5], learning_rate=[0.03, 0.08], reg_lambda=[1.0, 5.0]),
        # SVM: soft-margin C and RBF gamma
        "SVM (RBF)": grid(C=[0.3, 1.0, 3.0], gamma=["scale", 0.02]),
        # Decision Tree: depth, leaf size and cost-complexity pruning alpha
        "Decision Tree": grid(max_depth=[4, 6], min_samples_leaf=[15, 40], ccp_alpha=[0.0, 0.003]),
        # Logistic Regression: inverse L2 regularisation strength
        "Logistic Regression": grid(C=[0.05, 0.5, 5.0]),
        # Secondary candidates: one sensible configuration each
        "Extra Trees": [dict(max_depth=10, min_samples_leaf=5)],
        "Gaussian Process Classifier": [dict()],
    }


def generate_plots(
    test_probs: dict[str, np.ndarray],
    test_preds: dict[str, np.ndarray],
    yte: np.ndarray,
    Xtr_df: pd.DataFrame,
    Xte_df: pd.DataFrame,
    ebm_model: ExplainableBoostingClassifier,
    cat_model: CatBoostClassifier,
    feature_labels: dict[str, str],
    plots_dir: Path,
) -> dict:
    """Generate and save publication-quality evaluation and XAI plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    primary_models = [
        "Explainable Boosting Machine (EBM)",
        "CatBoost",
        "Random Forest",
        "XGBoost",
        "SVM (RBF)",
        "Decision Tree",
    ]
    colors = {
        "Explainable Boosting Machine (EBM)": "#D95F02",
        "CatBoost": "#1B9E77",
        "Random Forest": "#7570B3",
        "XGBoost": "#E7298A",
        "SVM (RBF)": "#66A61E",
        "Decision Tree": "#E6AB02",
    }

    # 1. ROC & Precision-Recall Curves + Calibration Curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for name in primary_models:
        if name not in test_probs:
            continue
        probs = test_probs[name]
        fpr, tpr, _ = roc_curve(yte, probs)
        prec, rec, _ = precision_recall_curve(yte, probs)
        auc_val = roc_auc_score(yte, probs)
        ap_val = average_precision_score(yte, probs)
        is_ebm = "Explainable Boosting Machine" in name
        lw = 2.8 if is_ebm else 1.8
        ls = "-" if is_ebm else "--"
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=colors[name], lw=lw, ls=ls)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap_val:.3f})", color=colors[name], lw=lw, ls=ls)

        prob_true, prob_pred = calibration_curve(yte, probs, n_bins=8)
        axes[2].plot(prob_pred, prob_true, marker="o", label=name, color=colors[name], lw=lw, ls=ls)

    axes[0].plot([0, 1], [0, 1], "k:", alpha=0.5)
    axes[0].set_title("ROC Curves (2018 Unseen Test Set)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right", fontsize=8.5)

    axes[1].axhline(yte.mean(), color="k", ls=":", alpha=0.5, label=f"Baseline ({yte.mean():.2f})")
    axes[1].set_title("Precision-Recall Curves (2018 Test Set)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", fontsize=8.5)

    axes[2].plot([0, 1], [0, 1], "k:", alpha=0.5, label="Perfect Calibration")
    axes[2].set_title("Probability Calibration (Reliability Curve)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Observed Fraction of Rain Days")
    axes[2].legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_roc_pr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Confusion Matrices for EBM vs Baseline Models
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, name in zip(axes.flat, primary_models):
        cm = confusion_matrix(yte, test_preds[name])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10.5, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Dry (<1mm)", "Rain (>=1mm)"])
        ax.set_yticklabels(["Dry (<1mm)", "Rain (>=1mm)"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12, fontweight="bold",
                )
    plt.suptitle("Confusion Matrices on 2018 Test Split (Validation-Tuned Thresholds)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrices.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. EBM Intrinsic Glass-Box Global Feature & Interaction Importance
    ebm_global = ebm_model.explain_global()
    ebm_data = ebm_global.data()
    term_names = ebm_data["names"]
    term_scores = np.array(ebm_data["scores"])
    order = np.argsort(term_scores)[-15:]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    pretty_names = []
    for idx_t in order:
        raw_name = term_names[idx_t]
        if " & " in raw_name:
            p1, p2 = raw_name.split(" & ")
            pretty_names.append(f"Interaction: {p1} × {p2}")
        else:
            pretty_names.append(feature_labels.get(raw_name, raw_name))
    bars = ax.barh(
        range(len(order)),
        term_scores[order],
        color=["#D95F02" if " & " in term_names[i] else "#2B5C8F" for i in order],
    )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(pretty_names, fontsize=9.5)
    ax.set_xlabel("Mean Absolute Log-Odds Contribution |f(x)|", fontsize=10.5)
    ax.set_title("Explainable Boosting Machine (EBM) — Exact Global Feature & Pairwise Interaction Importance", fontsize=12, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "ebm_global_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. EBM Additive Shape Functions f_i(x_i) for Top 6 Main Features
    main_indices = [i for i in np.argsort(term_scores)[::-1] if " & " not in term_names[i]][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ebm_shapes = {}
    for idx_t in range(len(term_names)):
        t_name = term_names[idx_t]
        t_data = ebm_global.data(idx_t)
        if t_data.get("type") == "univariate":
            edges = np.array(t_data["names"], dtype=float)
            x_mid = 0.5 * (edges[:-1] + edges[1:]) if len(edges) == len(t_data["scores"]) + 1 else edges[:len(t_data["scores"])]
            ebm_shapes[t_name] = {
                "x": x_mid.tolist(),
                "scores": np.array(t_data["scores"], dtype=float).tolist(),
                "upper": np.array(t_data.get("upper_bounds", t_data["scores"]), dtype=float).tolist(),
                "lower": np.array(t_data.get("lower_bounds", t_data["scores"]), dtype=float).tolist(),
            }

    for ax, idx_t in zip(axes.flat, main_indices):
        t_name = term_names[idx_t]
        sh = ebm_shapes[t_name]
        x_vals = np.array(sh["x"])
        y_vals = np.array(sh["scores"])
        u_vals = np.array(sh["upper"])
        l_vals = np.array(sh["lower"])
        if t_name in DISCRETE_FEATURES:
            cats = [str(int(round(v))) for v in x_vals]
            ax.bar(range(len(cats)), y_vals, color=["#D95F02" if v > 0 else "#2B5C8F" for v in y_vals],
                   yerr=[y_vals - l_vals, u_vals - y_vals], capsize=5, alpha=0.9)
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(["No (0)", "Yes (1)"] if t_name.startswith("monsoon") else cats)
        else:
            ax.step(x_vals, y_vals, where="mid", color="#D95F02", lw=2.2, label="Learned Shape f(x)")
            ax.fill_between(x_vals, l_vals, u_vals, step="mid", color="#D95F02", alpha=0.22, label="±1 SD Bagging Band")
        ax.axhline(0.0, color="black", ls="--", lw=0.9, alpha=0.6)
        ax.set_title(feature_labels.get(t_name, t_name), fontsize=10, fontweight="bold")
        ax.set_xlabel(t_name)
        ax.set_ylabel("Log-Odds Contribution")
    axes.flat[0].legend(loc="best", fontsize=8.5)
    plt.suptitle("EBM Glass-Box Learned Non-Linear Shape Functions f_i(x_i)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "ebm_shape_functions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5a. SHAP on the EBM — model-agnostic Permutation SHAP on P(Rain), 2018 Test Set
    feat_cols = list(Xte_df.columns)

    def ebm_rain_prob(X):
        return ebm_model.predict_proba(pd.DataFrame(X, columns=feat_cols))[:, 1]

    background = Xtr_df.sample(n=min(100, len(Xtr_df)), random_state=42)
    ebm_shap_explainer = shap.Explainer(ebm_rain_prob, shap.maskers.Independent(background, max_samples=100),
                                        algorithm="permutation", seed=42)
    ebm_shap = ebm_shap_explainer(Xte_df)
    fig = plt.figure(figsize=(10, 6.5))
    shap.summary_plot(ebm_shap.values, Xte_df.rename(columns=feature_labels), max_display=12, show=False)
    plt.title("SHAP (Permutation) on EBM — Impact on P(Rain), 2018 Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_ebm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5b. SHAP cross-model check: TreeSHAP on CatBoost
    explainer = shap.TreeExplainer(cat_model)
    shap_values = explainer.shap_values(Xte_df)
    fig = plt.figure(figsize=(10, 6.5))
    shap.summary_plot(
        shap_values,
        Xte_df.rename(columns=feature_labels),
        max_display=12,
        show=False,
    )
    plt.title("Cross-Model Check: TreeSHAP on CatBoost (log-odds), 2018 Test Set", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ebm_shap_rank = pd.Series(np.abs(ebm_shap.values).mean(axis=0), index=feat_cols).sort_values(ascending=False)
    cat_shap_rank = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_cols).sort_values(ascending=False)

    # 6. LIME Local Explanation Plot on a Representative 2018 Wet Day
    lime_explainer = LimeTabularExplainer(
        training_data=Xtr_df.values,
        feature_names=[feature_labels.get(c, c) for c in Xtr_df.columns],
        class_names=["Dry (<1mm)", "Rain (>=1mm)"],
        mode="classification",
        random_state=42,
    )
    wet_indices = np.where((yte == 1) & (test_probs["Explainable Boosting Machine (EBM)"] > 0.75))[0]
    sample_idx = int(wet_indices[0]) if len(wet_indices) > 0 else 0
    lime_exp = lime_explainer.explain_instance(
        Xte_df.iloc[sample_idx].values,
        ebm_model.predict_proba,
        num_features=10,
    )
    fig = lime_exp.as_pyplot_figure()
    fig.set_size_inches(10, 5.5)
    plt.title(
        f"LIME Surrogate Local Explanation — EBM Forecast on {Xte_df.index[sample_idx].strftime('%Y-%m-%d')} "
        f"(P(Rain) = {test_probs['Explainable Boosting Machine (EBM)'][sample_idx]:.1%})",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "lime_local_explanation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "ebm_shapes": ebm_shapes,
        "ebm_term_names": term_names,
        "ebm_term_scores": term_scores.tolist(),
        "shap_values_test": shap_values,
        "ebm_shap_values_test": ebm_shap.values,
        "ebm_shap_top": ebm_shap_rank.head(10).round(4).to_dict(),
        "catboost_shap_top": cat_shap_rank.head(10).round(4).to_dict(),
        "shap_base_value": float(np.squeeze(explainer.expected_value)),
        "sample_wet_date": Xte_df.index[sample_idx].strftime("%Y-%m-%d"),
    }


def main():
    print("=" * 78)
    print("RATMALANA NEXT-DAY RAINFALL (>= 1.0 mm) — MODEL BENCHMARK & XAI PIPELINE")
    print("=" * 78)

    df, feats, feature_labels = build_dataset(DATA_DIR)
    print(f"Dataset compiled: {len(df)} daily observations (2012-2018) | {len(feats)} features")
    print(f"Overall Wet-Day Rate (>= 1.0 mm): {df['target'].mean():.3f}")
    missing_pct = (df[feats].isna().mean() * 100).round(2)
    print("Remaining feature missingness (%):", missing_pct[missing_pct > 0].to_dict())
    print("Calendar-impossible dates dropped per file (e.g. 30 Feb):", INVALID_DATE_LOG)

    # Chronological Train (2012-2016) / Validation (2017) / Test (2018) Split
    train_df = df[df.index.year <= 2016].copy()
    val_df = df[df.index.year == 2017].copy()
    test_df = df[df.index.year == 2018].copy()
    trval_df = pd.concat([train_df, val_df])

    print(
        f"Chronological Split -> Train (2012-2016): {len(train_df)} | "
        f"Val (2017): {len(val_df)} | Test (2018): {len(test_df)}"
    )

    # Fit Median Imputer strictly on training split (2012-2016) for validation step
    imp_val = SimpleImputer(strategy="median")
    X_tr_val = pd.DataFrame(imp_val.fit_transform(train_df[feats]), index=train_df.index, columns=feats)
    X_va_val = pd.DataFrame(imp_val.transform(val_df[feats]), index=val_df.index, columns=feats)
    y_tr_val = train_df["target"].values.astype(int)
    y_va_val = val_df["target"].values.astype(int)

    # Fit Median Imputer on Train+Val (2012-2017) for final evaluation on unseen 2018 Test set
    imp_final = SimpleImputer(strategy="median")
    X_trval = pd.DataFrame(imp_final.fit_transform(trval_df[feats]), index=trval_df.index, columns=feats)
    X_test = pd.DataFrame(imp_final.transform(test_df[feats]), index=test_df.index, columns=feats)
    y_trval = trval_df["target"].values.astype(int)
    y_test = test_df["target"].values.astype(int)

    param_grids = get_param_grids()

    rows = []
    tuning_rows = []
    test_probs = {}
    test_preds = {}
    val_thresholds = {}
    best_params = {}
    trained_models = {}

    cv_folds = []
    for val_year in [2014, 2015, 2016, 2017]:
        f_tr = df[df.index.year < val_year]
        f_va = df[df.index.year == val_year]
        f_imp = SimpleImputer(strategy="median")
        cv_folds.append((
            val_year,
            pd.DataFrame(f_imp.fit_transform(f_tr[feats]), index=f_tr.index, columns=feats),
            f_tr["target"].values.astype(int),
            pd.DataFrame(f_imp.transform(f_va[feats]), index=f_va.index, columns=feats),
            f_va["target"].values.astype(int),
        ))
    print("Hyperparameter search: rolling-origin CV, validation years", [f[0] for f in cv_folds])

    for name, grid in param_grids.items():
        best_auc, best_p = -1.0, None
        for params in grid:
            fold_aucs = []
            for val_year, f_Xtr, f_ytr, f_Xva, f_yva in cv_folds:
                mdl_v = build_model(name, params)
                mdl_v.fit(f_Xtr, f_ytr)
                fold_aucs.append(roc_auc_score(f_yva, mdl_v.predict_proba(f_Xva)[:, 1]))
            mean_auc = float(np.mean(fold_aucs))
            tuning_rows.append({
                "model": name, "params": str(params), "CV_mean_ROC-AUC": mean_auc,
                "CV_std_ROC-AUC": float(np.std(fold_aucs)), "fold_ROC-AUCs": str(np.round(fold_aucs, 4).tolist()),
            })
            if mean_auc > best_auc:
                best_auc, best_p = mean_auc, params

        mdl_v = build_model(name, best_p)
        mdl_v.fit(X_tr_val, y_tr_val)
        best_val_probs = mdl_v.predict_proba(X_va_val)[:, 1]
        val_auc_2017 = roc_auc_score(y_va_val, best_val_probs)
        print(f"  {name:<38} CV mean ROC-AUC={best_auc:.4f} over {len(grid)} config(s) -> {best_p}")

        opt_t = find_optimal_threshold(y_va_val, best_val_probs)
        val_thresholds[name] = opt_t
        best_params[name] = best_p

        mdl_f = build_model(name, best_p)
        mdl_f.fit(X_trval, y_trval)
        p_test = mdl_f.predict_proba(X_test)[:, 1]
        yhat_test = (p_test >= opt_t).astype(int)

        test_probs[name] = p_test
        test_preds[name] = yhat_test
        trained_models[name] = mdl_f

        rows.append({
            "model": name,
            "CV_ROC-AUC": best_auc,
            "Val_ROC-AUC": val_auc_2017,
            "ROC-AUC": roc_auc_score(y_test, p_test),
            "PR-AUC": average_precision_score(y_test, p_test),
            "F1": f1_score(y_test, yhat_test),
            "Precision": precision_score(y_test, yhat_test, zero_division=0),
            "Recall": recall_score(y_test, yhat_test, zero_division=0),
            "Accuracy": accuracy_score(y_test, yhat_test),
            "Brier": brier_score_loss(y_test, p_test),
            "Opt_Threshold": opt_t,
            "Best_Params": str(best_p),
        })

    tuning_df = pd.DataFrame(tuning_rows).sort_values(["model", "CV_mean_ROC-AUC"], ascending=[True, False])
    tuning_df.to_csv(TUNING_CSV, index=False)
    print(f"Saved full hyperparameter search log to: {TUNING_CSV}")

    results_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print("\nBenchmark Results (Sorted by 2018 Test ROC-AUC):")
    print(results_df.drop(columns=["Best_Params"]).round(4).to_string(index=False))
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved benchmark metrics table to: {OUT_CSV}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    xai_artifacts = generate_plots(
        test_probs=test_probs,
        test_preds=test_preds,
        yte=y_test,
        Xtr_df=X_trval,
        Xte_df=X_test,
        ebm_model=trained_models["Explainable Boosting Machine (EBM)"],
        cat_model=trained_models["CatBoost"],
        feature_labels=feature_labels,
        plots_dir=PLOTS_DIR,
    )
    print(f"Saved evaluation & XAI plots to: {PLOTS_DIR}")
    print("EBM intrinsic top terms:", dict(sorted(zip(xai_artifacts["ebm_term_names"], np.round(xai_artifacts["ebm_term_scores"], 4)), key=lambda kv: -kv[1])[:10]))
    print("SHAP-on-EBM top features:", xai_artifacts["ebm_shap_top"])
    print("TreeSHAP-on-CatBoost top features:", xai_artifacts["catboost_shap_top"])
    print("LIME sample date:", xai_artifacts["sample_wet_date"])

    feature_stats = {}
    for col in feats:
        s = X_trval[col]
        feature_stats[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
        }

    bundle = {
        "models": trained_models,
        "imputer": imp_final,
        "features": feats,
        "feature_labels": feature_labels,
        "feature_stats": feature_stats,
        "val_thresholds": val_thresholds,
        "best_params": best_params,
        "invalid_dates": INVALID_DATE_LOG,
        "results_df": results_df,
        "X_trval": X_trval,
        "y_trval": y_trval,
        "X_test": X_test,
        "y_test": y_test,
        "test_target_mm": test_df["target_mm"].values,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "full_df": df,
        "xai": xai_artifacts,
    }
    bundle_path = ARTIFACTS_DIR / "models_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    print(f"Saved trained models & XAI bundle for frontend to: {bundle_path}")


if __name__ == "__main__":
    main()
