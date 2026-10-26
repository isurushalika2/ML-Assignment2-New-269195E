# Sri Lanka Next-Day Rainfall Prediction & Explainable AI (XAI) System

**Course:** Applied Machine Learning (MSc in AI)  
**Target Station:** Ratmalana Meteorological Station (Station ID: `43467`, Western Province, Sri Lanka)  
**Task:** Binary Classification — Predicting Next-Day Wet Days ($\text{Precipitation}_{t+1} \ge 1.0\text{ mm}$, WMO wet-day standard)  
**Featured Glass-Box Algorithm:** **Explainable Boosting Machine (EBM)** (`interpret.glassbox.ExplainableBoostingClassifier`)  
**Compared Baseline Algorithms:** Random Forest, CatBoost, XGBoost, Support Vector Machine (RBF Kernel), Decision Tree (CART), and Logistic Regression (plus secondary candidates Extra Trees and Gaussian Process Classifier)  
**Live App:** `[add your Streamlit Community Cloud URL here]`

---

## 1. Project Overview & Problem Definition

Accurate short-range daily rainfall forecasting in Sri Lanka's Western Province is critical for urban flood preparedness, aviation operations at Ratmalana Airport, municipal drainage management, and agricultural scheduling across the Colombo–Kalutara corridor. Because tropical convective and monsoonal rainfall exhibits strong non-linear thresholds and multi-station spatial dependencies, black-box models often struggle to justify their alerts to meteorologists and disaster-management authorities.

This project compiles multi-station daily meteorological records (2012–2018) from the **Department of Meteorology, Sri Lanka**, engineers leak-free spatial, temporal-persistence, and seasonal monsoon features, tunes and benchmarks the **Explainable Boosting Machine (EBM)** against standard supervised machine learning algorithms, interprets the learned meteorological relationships using **intrinsic EBM shape functions**, **SHAP**, and **LIME**, and deploys the trained models as an interactive **Streamlit web dashboard** (`app.py`).

---

## 2. Project Directory Structure

```text
MLAssignment2New_269195E/
├── data/                                    # Raw Sri Lanka Dept. of Meteorology CSV files (2012–2018)
│   ├── ratmalana_rf_2012-2018.csv           # Target station daily precipitation (mm)
│   ├── ratmalana_rh_2012-2018.csv           # Target station daily max relative humidity (%)
│   ├── ratmalana_t_2012-2018.csv            # Target station daily max temperature (°C)
│   ├── colombo_rh_2012-2018.csv             # Colombo synoptic station relative humidity (%)
│   ├── colombo_t_2012-2018.csv              # Colombo synoptic station temperature (°C)
│   ├── dehiwala_rf_2012-2018.csv            # Dehiwala Zoo rainfall + 2015 Homagama rainfall (mm)
│   ├── homagama_rf_2012-2018.csv            # Homagama inland station rainfall (mm)
│   └── ...                                  # Auxiliary 2017–2018 regional station files
├── artifacts/
│   ├── models_bundle.joblib                 # Trained models, imputer, thresholds, tuned params & XAI data (~24 MB, compressed)
│   └── plots/                               # Generated evaluation & XAI figures
│       ├── model_comparison_roc_pr.png      # ROC, Precision-Recall, and Calibration curves (2018 Test)
│       ├── confusion_matrices.png           # Confusion matrices across top models
│       ├── ebm_global_importance.png        # EBM global feature importance
│       ├── ebm_shape_functions.png          # EBM non-linear shape functions f_i(x_i) (bars for binary/count features)
│       ├── shap_summary_ebm.png             # SHAP (Permutation) beeswarm on the EBM itself
│       ├── shap_summary_beeswarm.png        # TreeSHAP beeswarm on CatBoost (cross-model check)
│       └── lime_local_explanation.png       # LIME local surrogate explanation for a wet-day forecast
├── model_selection_benchmark.py             # End-to-end preprocessing, tuning, training, evaluation & XAI script
├── model_selection_results.csv              # Final benchmark metrics (CV, Val & Test)
├── hyperparameter_tuning_results.csv        # Every hyperparameter configuration tried, with per-fold ROC-AUC
├── app.py                                   # Interactive 4-tab Streamlit frontend dashboard
├── main.py                                  # One-click PyCharm entry point to train (if needed) & launch the frontend
├── requirements.txt                         # Pinned Python dependencies (match the trained model bundle)
├── .gitignore                               # Keeps .venv, .idea, catboost_info, .DS_Store out of Git
└── README.md                                # Project documentation and execution guide
```

---

## 3. Dataset Description, Data Cleaning & Feature Engineering

### 3.1 Data Source & Coverage
The dataset consists of official daily meteorological observations from **2012-01-01 to 2018-12-31** ($N = 2,556$ valid daily records after aligning next-day targets, with an overall wet-day prevalence of **$40.6\%$** where $\text{Precipitation}_{t+1} \ge 1.0\text{ mm}$). Four stations with continuous 7-year coverage (2012–2018) across the Western Province are integrated:
1. **Ratmalana (`43467`)**: Daily precipitation (`PRECIP`), maximum relative humidity (`RHMAX`), and maximum temperature (`TMPMAX`).
2. **Colombo (`43466`)**: Daily maximum relative humidity (`RHMAX`) and maximum temperature (`TMPMAX`).
3. **Dehiwala Zoo**: Coastal neighboring daily precipitation (`PRECIP`).
4. **Homagama**: Inland neighboring daily precipitation (`PRECIP`).

*(Note: Auxiliary station files covering only 2017–2018 such as Angoda, Henarathgoda, Katunayake, Nittambuwa, and Welisara Navy are excluded from the 2012–2018 feature matrix to avoid $71\%$ structural missingness across the 2012–2016 training period.)*

### 3.2 Key Data Quality Fixes Implemented
1. **Calendar-Impossible Dates**: The raw exports contain non-existent dates such as `2012-02-30`, `2012-02-31` and `2012-04-31` (20 rows in each of the Ratmalana, Colombo and Homagama files, 18 in the Dehiwala file). They are parsed with `errors="coerce"`, counted, logged by the pipeline, and dropped.
2. **Recovery of Misplaced 2015 Homagama Rainfall**: `homagama_rf_2012-2018.csv` omits the entire year 2015, while all 365 days of 2015 `HOMAGAMA` precipitation are stored inside `dehiwala_rf_2012-2018.csv`. Merging both sources reduced `hom_rain` missingness from $\sim 16\%$ down to **$3.6\%$**.
3. **Sensor Dropout Filtering**: Four anomalous `0.0 °C` readings in `ratmalana_t_2012-2018.csv` (`2014-10-16`, `2015-01-04`, `2015-10-10`, `2016-04-05`) are physically impossible in coastal Sri Lanka and are converted to `NaN` (`rat_t[rat_t < 15.0] = np.nan`).
4. **Deduplication & Unrecorded Element Exclusion**: In `colombo_rh_2012-2018.csv`, every 2018 `RHMAX` day appears twice (730 rows for 365 days); duplicates are removed by date. `col_TMPMIN` (recorded only 2015–2017, so 0 rows in the 2018 test year) and `col_RHMIN` (recorded only 2015–2018) were excluded so the test set is never evaluated on 100%-imputed columns.
5. **Leak-Free Imputation**: Missing values are imputed using a `SimpleImputer(strategy="median")` fit **only on the training portion** of each split (including inside every cross-validation fold) before transforming the validation/test data.

### 3.3 Engineered Features (24 Features)
All features at index $t$ use observations strictly up to day $t$ to predict day $t+1$:
- **Same-Day Station Readings ($t$)**: `rat_rain`, `deh_rain`, `hom_rain`, `rat_rhmax`, `col_rhmax`, `rat_tmax`, `col_tmax`.
- **Spatial Synoptic Consensus & Gradients ($t$)**:
  - `regional_rain_mean`: Mean rainfall across Ratmalana, Dehiwala, and Homagama on day $t$.
  - `stations_raining`: Number of Western Province stations recording $\ge 1.0\text{ mm}$ rain on day $t$ ($0$ to $3$).
  - `rh_diff_rat_col` (`rat_rhmax - col_rhmax`) & `tmax_diff_rat_col` (`rat_tmax - col_tmax`): Local coastal-inland gradients.
- **Temporal Lag & Cumulative Persistence ($t-6 \dots t$)**:
  - `rat_rain_lag1`, `deh_rain_lag1`, `hom_rain_lag1`: Previous day's rainfall ($t-1$).
  - `rat_rain_3d`, `rat_rain_7d`, `deh_rain_3d`: 3-day and 7-day rolling cumulative rainfall.
  - `rat_rhmax_change1d`, `rat_tmax_change1d`: 24-hour change in humidity and maximum temperature.
- **Cyclical Seasonality & Sri Lanka Monsoon Regimes**:
  - `doy_sin`, `doy_cos`: Cyclical sine/cosine encoding of day-of-year.
  - `monsoon_SW` (May–Sep Southwest Monsoon), `monsoon_IM1` (Mar–Apr First Inter-Monsoon), `monsoon_IM2` (Oct–Nov Second Inter-Monsoon), with Dec–Feb Northeast Monsoon as the reference category (avoiding dummy variable multicollinearity).

---

## 4. Algorithm Selection: Why Explainable Boosting Machine (EBM)?

### 4.1 Standard Supervised Baseline Models
We benchmark against standard classical machine learning families across linear, kernel, single-tree, bagging, and boosting paradigms:
- **Linear & Generalized Linear Models**: Logistic Regression with L2 regularization.
- **Margin & Kernel Classifiers**: Support Vector Machine (SVM) with Radial Basis Function (RBF) kernel.
- **Tree & Ensemble Classifiers**: Decision Tree (CART with Cost-Complexity Pruning), Random Forest (Bootstrap Aggregating), XGBoost, and CatBoost.

### 4.2 Selected Glass-Box Algorithm: Explainable Boosting Machine (EBM)
We selected the **Explainable Boosting Machine (EBM)** (`ExplainableBoostingClassifier` from Microsoft Research's `interpret` library), along with two secondary comparison models (**Extra Trees** and **Gaussian Process Classifier**).

**How EBM Works & How It Differs from Standard Models:**
$$g(\mathbb{E}[y]) = \beta_0 + \sum_{i=1}^{p} f_i(x_i) \; \left[ + \sum_{(i,j) \in \mathcal{I}} f_{ij}(x_i, x_j) \right]$$
1. **Generalized Additive Model (GAM), optionally with pairwise interactions (GA²M)**: Unlike Logistic Regression (which assumes linear log-odds $\beta_i x_i$) and unlike XGBoost/CatBoost/Random Forest (which build trees where all features interact inside black-box paths), EBM learns one non-linear function per feature and adds them up. The bracketed interaction terms are optional; whether to use them was decided by hyperparameter search (Section 5.2).
2. **Cyclic (Round-Robin) Gradient Boosting**: EBM trains one feature function $f_i(x_i)$ at a time in a round-robin cycle using a small learning rate over thousands of iterations with outer bagging (`outer_bags=8`). Because only one feature is updated per step, the order of collinear features does not matter, and each variable gets a smooth non-linear lookup table.
3. **Automatic Pairwise Interaction Detection (FAST)**: When `interactions > 0`, EBM ranks all feature pairs on the residuals after the main effects and fits explicit 2D surfaces $f_{ij}(x_i, x_j)$ for the top $K$ pairs.
4. **Exact Intrinsic Glass-Box Interpretability**: Every prediction is the exact sum of the intercept $\beta_0$ plus the lookup values $f_i(x_i)$, so the model is its own explanation — no approximation is needed to see why a forecast was made.

---

## 5. Model Training, Hyperparameter Tuning & Benchmark Evaluation

### 5.1 Chronological Splitting Strategy
Because meteorological data is a time series with multi-day autocorrelation, a random `train_test_split` would leak future information into training. We use strictly chronological splits:
- **Training Split (`2012–2016`)**: $1,827$ days ($\sim 71.5\%$)
- **Validation Split (`2017`)**: $365$ days ($\sim 14.3\%$) — the last fold of the hyperparameter search, and used to tune each model's classification threshold $t^*$ (maximising F1).
- **Unseen Test Split (`2018`)**: $364$ days ($\sim 14.2\%$) — evaluated **once**, after refitting the selected configuration on `2012–2017`.

### 5.2 Hyperparameter Tuning (Rolling-Origin Time-Series Cross-Validation)
Each model family has a small grid over its main complexity / regularisation controls (`get_param_grids()` in `model_selection_benchmark.py`). Every configuration is scored by **rolling-origin (expanding-window) CV**: validate on 2014, 2015, 2016 and 2017 in turn, each time training only on the earlier years, and take the **mean ROC-AUC**. A single validation year (365 days) proved too noisy to separate configurations (the eight EBM configurations scored within 0.782–0.787 on 2017 alone). The full log of every configuration and fold is saved to `hyperparameter_tuning_results.csv`.

| Model | Grid searched | Selected configuration | CV mean ROC-AUC |
| :--- | :--- | :--- | :---: |
| **Explainable Boosting Machine (EBM)** | `learning_rate` {0.015, 0.04} × `interactions` {0, 10} × `min_samples_leaf` {10, 25} | `learning_rate=0.015, interactions=0, min_samples_leaf=25` | **0.745** |
| Random Forest | `max_depth` {6, 10, None} × `min_samples_leaf` {3, 10} (500 trees) | `max_depth=10, min_samples_leaf=10` | 0.748 |
| CatBoost | `depth` {4, 6} × `learning_rate` {0.03, 0.06} × `l2_leaf_reg` {3, 8} (500 iters) | `depth=4, learning_rate=0.03, l2_leaf_reg=8` | 0.734 |
| XGBoost | `max_depth` {3, 5} × `learning_rate` {0.03, 0.08} × `reg_lambda` {1, 5} (400 trees) | `max_depth=3, learning_rate=0.03, reg_lambda=5` | 0.738 |
| SVM (RBF) | `C` {0.3, 1, 3} × `gamma` {scale, 0.02} | `C=0.3, gamma=0.02` | 0.740 |
| Decision Tree | `max_depth` {4, 6} × `min_samples_leaf` {15, 40} × `ccp_alpha` {0, 0.003} | `max_depth=4, min_samples_leaf=15, ccp_alpha=0.003` | 0.726 |
| Logistic Regression | `C` {0.05, 0.5, 5} | `C=0.05` | 0.730 |

**What tuning showed:** nearly every model preferred its **more regularised** options (shallow trees, slow learning rates, strong L2, larger leaves), which is expected with only ~1,800 training days. For EBM, **all four configurations without interactions beat all four with 10 pairwise interactions** — the interaction terms overfit, so the selected model is a pure additive GAM. This also makes it maximally interpretable: every feature's effect is a single 1-D curve.

### 5.3 Benchmark Results

| Model | Category | CV ROC-AUC (2014–17) | Val ROC-AUC (2017) | **Test ROC-AUC (2018)** | Test PR-AUC | Test F1 | Test Precision | Test Recall | Test Accuracy | Test Brier | Threshold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Random Forest` | Baseline (Bagging) | **0.748** | 0.789 | **0.799** | **0.708** | 0.657 | **0.669** | 0.645 | **0.739** | **0.178** | 0.500 |
| `Extra Trees` | Secondary Candidate | 0.748 | 0.781 | 0.795 | 0.685 | **0.678** | 0.632 | 0.730 | 0.731 | 0.179 | 0.400 |
| `CatBoost` | Baseline (Boosting) | 0.734 | 0.788 | 0.795 | 0.695 | 0.669 | 0.593 | **0.766** | 0.706 | 0.180 | 0.360 |
| `XGBoost` | Baseline (Boosting) | 0.738 | 0.781 | 0.788 | 0.677 | 0.615 | 0.643 | 0.589 | 0.714 | 0.184 | 0.515 |
| `Gaussian Process Classifier` | Secondary Candidate | 0.742 | 0.780 | 0.788 | 0.662 | 0.673 | 0.647 | 0.702 | 0.736 | 0.180 | 0.440 |
| **`Explainable Boosting Machine (EBM)`** | **Primary Glass-Box Model** | **0.745** | 0.786 | **0.778** | 0.647 | 0.660 | 0.634 | 0.688 | 0.725 | 0.184 | 0.425 |
| `SVM (RBF)` | Baseline (Kernel) | 0.740 | 0.776 | 0.776 | 0.646 | 0.662 | 0.632 | 0.695 | 0.725 | 0.185 | 0.385 |
| `Logistic Regression` | Baseline (Linear) | 0.730 | 0.778 | 0.770 | 0.644 | 0.676 | 0.651 | 0.702 | 0.739 | 0.186 | 0.435 |
| `Decision Tree` | Baseline (CART) | 0.726 | 0.776 | 0.766 | 0.634 | 0.652 | 0.584 | 0.738 | 0.695 | 0.188 | 0.345 |

**Key Takeaways:**
- **Across four validation years**, EBM (CV ROC-AUC **0.745**) is within **0.003** of the best model (Random Forest, 0.748) and **ahead of CatBoost, XGBoost, SVM, Logistic Regression and the Decision Tree** — i.e. a fully transparent additive model matches the strongest black-box ensembles on average.
- **On the single 2018 test year**, EBM (0.778) trails Random Forest (0.799) and CatBoost (0.795) by about **0.02 ROC-AUC** while still beating SVM, Logistic Regression and the Decision Tree. With 364 test days, gaps of this size are close to sampling noise; we report them honestly as the accuracy cost of full interpretability.
- Precision/recall depend on each model's validation-tuned threshold: e.g. CatBoost's low threshold (0.36) gives the highest recall (76.6%) at the cost of precision (59.3%). For a flood-warning use case a lower threshold (more wet days caught, more false alarms) is the safer operating point.
- **Robustness note:** with a single-year (2017) hyperparameter search instead of rolling-origin CV, the selected EBM configuration scored 0.778 on 2018 as well, and the untuned default configuration 0.791. We report the CV-selected model because choosing a configuration by its test score would leak the test set.

---

## 6. Explainability & Interpretation (XAI) Findings

Three complementary XAI methods are applied to the EBM and saved to `artifacts/plots/`:

1. **EBM Intrinsic Global Importance & Shape Functions (`ebm_global_importance.png`, `ebm_shape_functions.png`)**:
   - The top global drivers of next-day rainfall at Ratmalana are **Ratmalana rainfall today (`rat_rain`, mean $|f| = 0.252$)**, **number of stations raining today (`stations_raining`, 0.174)**, **Homagama rainfall today (`hom_rain`, 0.148)**, **Southwest Monsoon (`monsoon_SW`, 0.109)**, **regional mean rainfall (`regional_rain_mean`, 0.109)**, **7-day cumulative rain (`rat_rain_7d`, 0.105)** and the **Second Inter-Monsoon (`monsoon_IM2`, 0.090)**, followed by maximum relative humidity at Ratmalana (0.079) and Colombo (0.078).
   - **Alignment with domain knowledge:** a completely dry day (`rat_rain = 0`, `stations_raining = 0`) pushes the log-odds strongly negative ($\approx -0.26$ and $-0.21$), while rain at 2–3 stations, high 7-day totals and the SW and second inter-monsoon seasons push it positive — matching the persistence of monsoon rain spells over the Western Province. Higher maximum humidity at both stations also raises the rain probability.
   - **A counter-intuitive shape, explained:** the `rat_rain` curve rises up to about 40 mm and then **drops sharply** for heavier rain. This is real in the data, not an artefact: in 2012–2017 the next day was wet after **71%** of days with 10–40 mm, but after only **59%** of the **85** days with ≥ 40 mm (heavy convective storms are often followed by clearing). Two further reasons: (i) only 85 training days are that wet, so the band is wide and the estimate uncertain; (ii) in an additive model credit is *shared* between correlated features — 50 of those 85 days also had rain at all three stations, so part of the heavy-rain signal is carried by `stations_raining` and `regional_rain_mean` instead.
2. **SHAP (`shap_summary_ebm.png`, `shap_summary_beeswarm.png`)**:
   - **SHAP on the EBM itself** (model-agnostic Permutation SHAP on P(Rain), 100-day background sample) ranks the **same top three features in the same order** as EBM's intrinsic importance (`rat_rain`, `stations_raining`, `hom_rain`), and the same top-ten set, confirming the glass-box explanation is faithful.
   - **TreeSHAP on CatBoost** (cross-model check) agrees that same-day rainfall at Ratmalana and Homagama, station agreement, 7-day persistence and humidity dominate, but gives more weight to seasonality (`doy_sin`) — a black-box model can spread the same seasonal signal over different features.
3. **LIME & EBM Local Explanations (`lime_local_explanation.png` & Dashboard Tab 1)**:
   - For an individual forecast (e.g. `2018-05-19`, where EBM predicts $P(\text{Rain}) = 75.6\%$), LIME's perturbation surrogate identifies `hom_rain > 5.7 mm`, `rat_rain > 5.6 mm`, `stations_raining > 2`, the SW monsoon, `rat_rain_7d > 71.8 mm` and Colombo humidity > 91% as the main positive drivers, consistent with EBM's exact local breakdown shown in the dashboard.

---

## 7. Critical Discussion: Limitations, Data Quality & Ethical Impact

- **Model & Meteorological Limitations**: Surface station observations lack upper-air variables (e.g. 850 hPa winds, pressure tendency, satellite cloud-top temperatures). Sudden convective storms after completely dry days therefore remain hard to predict from surface gauges alone (test ROC-AUC $\approx 0.78$–$0.80$ for all strong models). Only seven years of data are available, so year-to-year variation is large: the selected EBM's ROC-AUC ranges from 0.72 (2015) to 0.79 (2017) across CV folds, and Random Forest's from 0.72 to 0.79.
- **Evaluation Limitations**: The test set is a single year (364 days). Differences of about ±0.02 ROC-AUC between the top models are within year-to-year noise, so the ranking among the top 6 models should not be over-interpreted.
- **Data Quality Challenges**: Calendar-impossible dates, recording gaps in minimum temperature (`TMPMIN`) and minimum humidity (`RHMIN`), occasional sensor dropouts (`0.0 °C`), duplicated rows and misfiled station records (2015 Homagama data inside the Dehiwala file) all required auditing before modeling.
- **Fairness, Bias & Ethical Considerations**: The dataset contains only non-personal environmental sensor measurements (no privacy risk). However, the gauges are concentrated in urban/suburban Western Province centers (Colombo, Ratmalana, Dehiwala, Homagama), so applying the model to rural or highland districts (e.g. Ratnapura or Nuwara Eliya) without local retraining would introduce geographic representation bias. False negatives during heavy monsoon periods carry flood-preparedness risks; the tool is a decision aid and must not replace official Department of Meteorology forecasts.

---

## 8. Step-by-Step Instructions to Run the Project

### Option A: Run Directly in PyCharm (Recommended)
The local virtual environment (`.venv`) has all packages installed, and pre-trained artifacts are saved in `artifacts/`.

1. **Run the Interactive Frontend Dashboard (`main.py`)**:
   - Open `main.py` in PyCharm and click the green **Run (`⌃R`)** button.
   - `main.py` checks for `artifacts/models_bundle.joblib` (running `model_selection_benchmark.py` if it is missing) and launches the **Streamlit dashboard** at **`http://localhost:8501`**.
2. **Re-run the Full Tuning, Benchmark & XAI Pipeline (`model_selection_benchmark.py`)**:
   - Open `model_selection_benchmark.py` and click **Run (`⌃R`)** (takes about 5 minutes).
   - It rebuilds the dataset from `data/`, runs the rolling-origin hyperparameter search, trains all 9 models, prints the benchmark table, updates `model_selection_results.csv` and `hyperparameter_tuning_results.csv`, regenerates all 7 PNG figures, and updates `artifacts/models_bundle.joblib`.

### Option B: Run from the Terminal (macOS / Linux)

```bash
# 1. (Optional) install the pinned dependencies into .venv
./.venv/bin/pip install -r requirements.txt

# 2. Tuning, training, benchmark & XAI pipeline
./.venv/bin/python model_selection_benchmark.py

# 3. Launch the Streamlit frontend, then open http://localhost:8501
./.venv/bin/streamlit run app.py
```

---

## 9. Deployment (Bonus): Streamlit Community Cloud

The app is deployable as-is: the trained bundle is committed, so the cloud app starts without retraining, and `requirements.txt` pins the exact library versions used to train it (tested on Python 3.9 and 3.11).

1. **Create the GitHub repository** (in PyCharm Terminal, inside the project folder):
   ```bash
   git init
   git add .
   git commit -m "Ratmalana next-day rainfall prediction with EBM and XAI dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
   `.gitignore` keeps `.venv/`, `.idea/`, `catboost_info/` and `.DS_Store` out of the repository.
2. **Deploy:** go to **share.streamlit.io** → sign in with GitHub → **Create app** → choose the repository, branch `main`, main file `app.py` → **Advanced settings → Python 3.11** → **Deploy**.
3. **Copy the public URL** (e.g. `https://<app-name>.streamlit.app`) into the *Live App* line at the top of this README and into your report.
4. **Before recording the demo video**, open the URL once — a sleeping app takes about 30–60 seconds to wake up.

---

## 10. Guide to Using the Frontend Dashboard (For the Demo Video)

1. **Sidebar (`🎛️ Weather Input Controls`)**:
   - Choose **`📅 Pick a 2018 Test Date`** to select any real day from 2018 and compare the prediction with the actual observed next-day rainfall (mm), **or** tick **`Override specific features`** / choose **`🎚️ Custom Weather Scenario Sliders`** for live *what-if* experiments with rainfall, humidity, temperature and day of year (the selected date's season is kept when overriding; derived features such as station counts, 3-day totals and 24-hour changes are recomputed automatically).
2. **Tab 1 (`🔮 Live Forecast & Local XAI`)**:
   - Live next-day rain probability from **EBM** and **Random Forest**, a **multi-model consensus table** across all 9 models, and side-by-side **EBM exact local contributions** and **LIME surrogate weights** explaining that specific forecast.
3. **Tab 2 (`📊 Model Benchmark Comparison`)**:
   - The full CV / validation / 2018 test metrics table with the selected hyperparameters, an explanation of how to read it, plus ROC, Precision-Recall, calibration curves and confusion matrices.
4. **Tab 3 (`🧠 Global Explainability`)**:
   - EBM global importance, SHAP on EBM and the CatBoost TreeSHAP cross-check, plus an **interactive EBM shape-function explorer** showing any feature's learned curve $f_i(x_i)$ (bar chart for binary/count features) with your current input marked.
5. **Tab 4 (`🗺️ Dataset Explorer`)**:
   - Dataset summary metrics, multi-station daily rainfall time series, and the engineered feature matrix.
