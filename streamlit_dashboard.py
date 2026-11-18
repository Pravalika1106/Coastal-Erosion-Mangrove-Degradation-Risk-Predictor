import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
    r2_score,
    mean_absolute_error,
)

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Coastal Erosion & Mangrove Risk Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

ECO_FEATURES = [
    "abs_latitude",
    "dist_to_nearest_mangrove_km",
    "nearest_mangrove_patch_ha",
    "local_5patch_mangrove_sum_ha",
    "local_mangrove_density_index",
    "country_loss_pct_1996_2020",
    "country_recent_trend_2015_2020_ha",
]


# ==============================================================================
# LOAD DATA & TRAIN ALL 8 MODELS ONCE (CACHED FOR INSTANT UPDATES)
# ==============================================================================
@st.cache_resource
def load_and_train_all():
    df = pd.read_csv("final_coastal_mangrove_preprocessed.csv")
    df_sub = df.sample(n=min(6000, len(df)), random_state=42).reset_index(drop=True)

    X = df_sub[ECO_FEATURES]
    y_clf = df_sub["TARGET_CLF_Severe_Erosion"]
    y_reg = df_sub["TARGET_REG_Shoreline_Change_m"]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_clf, y_reg, test_size=0.20, random_state=42, stratify=y_clf
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=42
        ),
        "Decision Tree (Unpruned)": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
        "Decision Tree (Pruned Depth=5)": DecisionTreeClassifier(
            max_depth=5, class_weight="balanced", random_state=42
        ),
        "Support Vector Machine": SVC(
            class_weight="balanced", probability=True, random_state=42
        ),
        "Multi-Layer Perceptron": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, random_state=42
        ),
        "Bagging Classifier": BaggingClassifier(n_estimators=50, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=42
        ),
    }

    trained_clfs = {}
    for name, m in models.items():
        m.fit(X_train_sc, yc_train)
        trained_clfs[name] = m

    reg = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=8, random_state=42
    )
    reg.fit(X_train_sc, yr_train)

    rf_clf = trained_clfs["Random Forest"]
    reg_preds = reg.predict(X_test_sc)
    global_metrics = {
        "rf_acc": accuracy_score(yc_test, rf_clf.predict(X_test_sc)),
        "rf_auc": roc_auc_score(yc_test, rf_clf.predict_proba(X_test_sc)[:, 1]),
        "reg_mae": mean_absolute_error(yr_test, reg_preds),
        "reg_r2": r2_score(yr_test, reg_preds),
        "importances": pd.Series(
            rf_clf.feature_importances_, index=ECO_FEATURES
        ).sort_values(ascending=True),
    }
    return (
        df,
        scaler,
        trained_clfs,
        reg,
        X_test_sc,
        yc_test.values,
        global_metrics,
    )


(
    df_global,
    scaler,
    trained_clfs,
    reg,
    X_test_sc,
    yc_test,
    global_metrics,
) = load_and_train_all()

# ==============================================================================
# SIDEBAR ONLY: SINGLE CONTROL PANEL FOR ALL TABS
# ==============================================================================
st.sidebar.header("⚡ Quick Scenario Presets")
preset = st.sidebar.radio(
    "Load Coastal Profile:",
    [
        "Custom Slider Values",
        "Healthy Dense Mangrove Buffer",
        "Degraded / Distant Coastline",
    ],
)

if preset == "Healthy Dense Mangrove Buffer":
    def_lat, def_dist, def_patch, def_sum5, def_loss, def_trend = (
        22.5,
        0.0,
        450.0,
        1900.0,
        3.5,
        2500.0,
    )
elif preset == "Degraded / Distant Coastline":
    def_lat, def_dist, def_patch, def_sum5, def_loss, def_trend = (
        8.5,
        45.0,
        1.5,
        8.0,
        -12.5,
        -16000.0,
    )
else:
    def_lat, def_dist, def_patch, def_sum5, def_loss, def_trend = (
        14.0,
        6.0,
        55.0,
        260.0,
        -1.5,
        -900.0,
    )

st.sidebar.markdown("---")
st.sidebar.header("🌊 Coastal & Mangrove Parameters")

abs_lat = st.sidebar.slider(
    "Absolute Latitude (° from Equator)", 0.0, 35.0, float(def_lat), step=0.5
)
dist_km = st.sidebar.slider(
    "Distance to Nearest Mangrove (km)", 0.0, 50.0, float(def_dist), step=0.5
)
patch_ha = st.sidebar.slider(
    "Nearest Mangrove Patch Area (ha)", 0.1, 500.0, float(def_patch), step=1.0
)
sum5_ha = st.sidebar.slider(
    "Local 5-Patch Mangrove Total (ha)", 1.0, 2000.0, float(def_sum5), step=5.0
)

density_idx = sum5_ha / (dist_km + 1.0)
st.sidebar.info(f"**Computed Mangrove Density Index:** `{density_idx:.2f}`")

country_loss_pct = st.sidebar.slider(
    "National Mangrove Net Change 1996–2020 (%)",
    -15.0,
    10.0,
    float(def_loss),
    step=0.1,
)
recent_trend_ha = st.sidebar.slider(
    "Recent National Trend 2015–2020 (ha)",
    -20000.0,
    10000.0,
    float(def_trend),
    step=100.0,
)

# ==============================================================================
# LIVE PREDICTION ENGINE
# ==============================================================================
st.title("🌿 Coastal Erosion & Mangrove Degradation Risk Predictor")
st.markdown(
    "Move the **sidebar sliders** to dynamically update the risk gauge, shoreline projection, and global comparison."
)

user_input = np.array(
    [
        [
            abs_lat,
            dist_km,
            patch_ha,
            sum5_ha,
            density_idx,
            country_loss_pct,
            recent_trend_ha,
        ]
    ]
)
user_scaled = scaler.transform(user_input)

rf_clf = trained_clfs["Random Forest"]
severe_prob = rf_clf.predict_proba(user_scaled)[0][1] * 100.0
pred_retreat_m = reg.predict(user_scaled)[0]


def render_kpi_card(title, main_val, badge_text, status="good"):
    styles = {
        "good": ("rgba(33, 195, 84, 0.16)", "#21c354", "↑"),
        "warn": ("rgba(255, 165, 0, 0.18)", "#ffa500", "→"),
        "bad": ("rgba(255, 75, 75, 0.16)", "#ff4b4b", "↓"),
        "info": ("rgba(31, 119, 180, 0.18)", "#4da6ff", "●"),
    }
    bg_color, txt_color, arrow = styles[status]
    return f"""
    <div style="padding: 4px 0;">
        <div style="font-size: 0.88rem; opacity: 0.85; margin-bottom: 4px;">{title}</div>
        <div style="font-size: 2.15rem; font-weight: 700; line-height: 1.15; margin-bottom: 8px;">{main_val}</div>
        <span style="background-color: {bg_color}; color: {txt_color}; padding: 3px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;">
            {arrow} {badge_text}
        </span>
    </div>
    """


if severe_prob < 49.5:
    risk_label, risk_status = "Low Erosion Risk", "good"
elif severe_prob <= 52.5:
    risk_label, risk_status = "Moderate Erosion Risk", "warn"
else:
    risk_label, risk_status = "High Erosion Risk", "bad"

if pred_retreat_m >= -10.0:
    shore_label, shore_status = (
        "Accretion / Gain" if pred_retreat_m >= 0 else "Stable / Minor Shift",
        "good",
    )
elif pred_retreat_m >= -25.0:
    shore_label, shore_status = "Moderate Retreat", "warn"
else:
    shore_label, shore_status = "Severe Retreat", "bad"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        render_kpi_card(
            "Severe Erosion Risk Probability",
            f"{severe_prob:.1f}%",
            risk_label,
            risk_status,
        ),
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        render_kpi_card(
            "Projected 2050 Shoreline Change",
            f"{pred_retreat_m:.2f} m",
            shore_label,
            shore_status,
        ),
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        render_kpi_card(
            "Random Forest AUC-ROC",
            f"{global_metrics['rf_auc']:.3f}",
            f"Accuracy: {global_metrics['rf_acc']*100:.1f}%",
            "info",
        ),
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        render_kpi_card(
            "Regression MAE",
            f"{global_metrics['reg_mae']:.2f} m",
            f"R² Score: {global_metrics['reg_r2']:.3f}",
            "info",
        ),
        unsafe_allow_html=True,
    )

st.write("")

if risk_status == "bad":
    st.error(
        f"⚠️ **HIGH VULNERABILITY ALERT:** Simulated coastline has a **{severe_prob:.1f}% probability** of severe erosion "
        f"(projected **{pred_retreat_m:.1f} m** retreat by 2050)."
    )
elif risk_status == "warn":
    st.warning(
        f"⚡ **MODERATE VULNERABILITY:** Simulated coastline shows a **{severe_prob:.1f}% probability** of severe erosion "
        f"(projected **{pred_retreat_m:.1f} m** shoreline change by 2050). Strengthening mangrove buffer density is advised."
    )
else:
    st.success(
        f"✅ **PROTECTED / STABLE BUFFER:** Simulated coastline has a low **{severe_prob:.1f}% probability** of severe erosion "
        f"(projected **{pred_retreat_m:.1f} m** shoreline change by 2050)."
    )

st.markdown("---")

# ==============================================================================
# TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Live Risk Gauge & Sensitivity",
        "🌍 Global Mangrove-Shoreline Explorer",
        "📊 Model Benchmarks",
    ]
)

# --- TAB 1: GAUGE + FEATURE IMPORTANCE + WHAT-IF CURVE ---
with tab1:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        gauge_color = (
            "#d62728"
            if risk_status == "bad"
            else ("#ffa500" if risk_status == "warn" else "#2ca02c")
        )
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=severe_prob,
                title={"text": "Severe Erosion Risk Score (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 49.5], "color": "#e8f5e9"},
                        {"range": [49.5, 52.5], "color": "#fff3e0"},
                        {"range": [52.5, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 52.5,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with c2:
        imp_df = global_metrics["importances"].reset_index()
        imp_df.columns = ["Feature", "Importance"]
        fig_imp = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance",
            color="Importance",
            color_continuous_scale="Tealgrn",
        )
        fig_imp.update_layout(
            height=340, margin=dict(l=20, r=20, t=50, b=20), showlegend=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Sensitivity curve reacting to sidebar parameters
    dist_grid = np.linspace(0, 50, 50)
    whatif_rows = np.column_stack(
        [
            np.full_like(dist_grid, abs_lat),
            dist_grid,
            np.full_like(dist_grid, patch_ha),
            np.full_like(dist_grid, sum5_ha),
            sum5_ha / (dist_grid + 1.0),
            np.full_like(dist_grid, country_loss_pct),
            np.full_like(dist_grid, recent_trend_ha),
        ]
    )
    whatif_probs = rf_clf.predict_proba(scaler.transform(whatif_rows))[:, 1] * 100.0

    fig_whatif = go.Figure()
    fig_whatif.add_trace(
        go.Scatter(
            x=dist_grid,
            y=whatif_probs,
            mode="lines",
            name="Risk Curve",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig_whatif.add_trace(
        go.Scatter(
            x=[dist_km],
            y=[severe_prob],
            mode="markers",
            name="Your Sidebar Position",
            marker=dict(color="red", size=15, symbol="star"),
        )
    )
    fig_whatif.update_layout(
        title="Mangrove Buffer Distance vs. Severe Erosion Risk",
        xaxis_title="Distance to Nearest Mangrove (km)",
        yaxis_title="Severe Erosion Probability (%)",
        height=310,
    )
    st.plotly_chart(fig_whatif, use_container_width=True)

# --- TAB 2: GLOBAL SCATTER + MOVING SIDEBAR STAR ---
with tab2:
    sample_map = df_global.sample(
        n=min(2000, len(df_global)), random_state=42
    )
    st.caption(
        "Global JRC & Mangrove dataset (2,000 sampled transects) with your simulated coastline plotted as ★:"
    )

    fig_scatter = px.scatter(
        sample_map,
        x="dist_to_nearest_mangrove_km",
        y="TARGET_REG_Shoreline_Change_m",
        color="C_ID",
        hover_data=[
            "lat",
            "lon",
            "local_mangrove_density_index",
            "country_loss_pct_1996_2020",
        ],
        labels={
            "dist_to_nearest_mangrove_km": "Distance to Nearest Mangrove (km)",
            "TARGET_REG_Shoreline_Change_m": "Projected 2050 Shoreline Change (m)",
            "C_ID": "Region",
        },
        opacity=0.60,
    )

    fig_scatter.add_trace(
        go.Scatter(
            x=[dist_km],
            y=[pred_retreat_m],
            mode="markers+text",
            name="★ Your Simulated Coastline",
            text=[f"★ You ({pred_retreat_m:.1f}m)"],
            textposition="top center",
            marker=dict(
                color="red",
                size=18,
                symbol="star",
                line=dict(width=2, color="black"),
            ),
        )
    )

    fig_scatter.add_hline(
        y=-25.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Severe Erosion Threshold (-25m)",
    )
    fig_scatter.update_layout(height=470)
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: CLEAN MODEL BENCHMARK TABLE ONLY ---
with tab3:
    st.caption(
        f"Evaluation metrics across all 8 classification models on the held-out test set ({len(yc_test)} transects) "
        f"alongside live predictions for your current sidebar parameters:"
    )

    rows = []
    for name, model in trained_clfs.items():
        user_prob = model.predict_proba(user_scaled)[0][1] * 100.0
        user_pred_label = (
            "Severe Erosion" if user_prob > 52.5 else "Stable / Moderate"
        )

        test_preds = model.predict(X_test_sc)
        test_probs = model.predict_proba(X_test_sc)[:, 1]

        rows.append(
            {
                "Model": name,
                "Your Input Risk (%)": round(user_prob, 1),
                "Your Input Verdict": user_pred_label,
                "Test Accuracy": round(accuracy_score(yc_test, test_preds), 4),
                "Test F1-Score": round(
                    f1_score(yc_test, test_preds, zero_division=0), 4
                ),
                "Test AUC-ROC": round(roc_auc_score(yc_test, test_probs), 4),
                "Test Log-Loss": round(
                    log_loss(yc_test, test_probs, labels=[0, 1]), 4
                ),
            }
        )

    live_bench_df = pd.DataFrame(rows)
    st.dataframe(live_bench_df, use_container_width=True)