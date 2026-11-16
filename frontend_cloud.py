import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FFF PLA Surface Roughness Predictor & Optimizer",
    page_icon="⚙️",
    layout="wide",
)

FEATURE_COLS = [
    "Angle",
    "Nozzle_Diameter",
    "Layer_Height",
    "Wall_Thickness",
    "Top_Thickness",
    "Infill_Density",
    "Infill_Pattern",
    "Extrusion_Temp",
    "Bed_Temp",
    "Print_Speed",
    "Fan_Speed",
]

PATTERN_MAP = {"Cubic": 0, "Grid": 1, "Triangle": 2}
INV_PATTERN_MAP = {0: "Cubic", 1: "Grid", 2: "Triangle"}


# --- LOAD DATA & MODELS (SUPPORTS BOTH PRE-SAVED .PKL AND AUTO-TRAINING ON CLOUD) ---
@st.cache_data
def load_dataset():
    if not os.path.exists("am_data.xlsx"):
        return None
    xls = pd.ExcelFile("am_data.xlsx")
    df = pd.read_excel("am_data.xlsx", sheet_name=xls.sheet_names[0])
    df.columns = FEATURE_COLS + ["Surface_Roughness"]
    return df


@st.cache_resource
def load_or_train_models():
    df = load_dataset()

    # If serialized models already exist in the repository, load them directly
    if os.path.exists("all_models.pkl") and os.path.exists("metrics.pkl"):
        trained_models = joblib.load("all_models.pkl")
        metrics_data = joblib.load("metrics.pkl")
        metrics_df = pd.DataFrame(metrics_data)
        return trained_models, metrics_df, df

    if df is None:
        raise FileNotFoundError(
            "Neither 'all_models.pkl' nor 'am_data.xlsx' was found in the app directory."
        )

    df_clean = df.copy()
    df_clean["Infill_Pattern"] = (
        df_clean["Infill_Pattern"].astype(str).str.strip().map(PATTERN_MAP)
    )

    X = df_clean[FEATURE_COLS]
    y = df_clean["Surface_Roughness"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_defs = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "KNN (K-Nearest Neighbors)": KNeighborsRegressor(n_neighbors=5),
    }

    trained_models = {}
    metrics_rows = []
    for name, model in model_defs.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        trained_models[name] = model
        metrics_rows.append(
            {
                "Model": name,
                "R2_Score": round(float(r2_score(y_test, preds)), 4),
                "MAE_um": round(float(mean_absolute_error(y_test, preds)), 4),
                "RMSE_um": round(float(root_mean_squared_error(y_test, preds)), 4),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    return trained_models, metrics_df, df


def get_predictions(input_df: pd.DataFrame, selected_algo: str, trained_models: dict):
    """Uses local FastAPI backend if running on localhost, otherwise uses embedded cloud engine."""
    payload = input_df.iloc[0].to_dict()
    payload["Model_Name"] = selected_algo
    try:
        resp = requests.post(
            "http://127.0.0.1:8000/predict", json=payload, timeout=0.6
        )
        if resp.status_code == 200:
            res = resp.json()
            return res["predicted_ra"], res["all_predictions"], "FastAPI REST Backend"
    except Exception:
        pass

    all_preds = {
        name: round(float(model.predict(input_df)[0]), 3)
        for name, model in trained_models.items()
    }
    selected_ra = all_preds[selected_algo]
    return selected_ra, all_preds, "Embedded Cloud Inference Engine"


# --- INITIALIZE MODELS ---
trained_models, metrics_df, raw_df = load_or_train_models()

# --- HEADER ---
st.title("⚙️ FFF PLA Surface Roughness ($R_a$) Predictor & Process Optimizer")
st.markdown(
    "**Data Science in Manufacturing Project** | **Om Kiran Ghodake** (Roll No. `26MEM3R06`, M.Tech CIM)  \n"
    "*Department of Mechanical Engineering, National Institute of Technology, Warangal*"
)
st.divider()

# --- SIDEBAR: PROCESS PARAMETERS ---
st.sidebar.header("1. Select ML Algorithm")
model_names = list(trained_models.keys())
default_idx = model_names.index("Random Forest") if "Random Forest" in model_names else 0
selected_algo = st.sidebar.selectbox(
    "Active Regression Model:", options=model_names, index=default_idx
)

st.sidebar.header("2. CAD & Slicing Parameters")
angle = st.sidebar.slider("Sloped Surface Angle (Degrees)", 10.0, 80.0, 30.0, step=5.0)
nozzle_diameter = st.sidebar.selectbox("Nozzle Diameter (mm)", [0.25, 0.40], index=1)
layer_height = st.sidebar.select_slider(
    "Layer Height (mm)", options=[0.10, 0.15, 0.20], value=0.20
)
wall_thickness = st.sidebar.select_slider(
    "Wall Thickness (mm)", options=[0.75, 1.00, 1.25], value=1.00
)
top_thickness = st.sidebar.select_slider(
    "Top Thickness (mm)", options=[1.20, 1.40, 1.60], value=1.40
)

st.sidebar.header("3. Infill & Thermal/Kinematic Settings")
infill_density = st.sidebar.select_slider(
    "Infill Density (%)", options=[20, 50, 80], value=50
)
infill_pattern_str = st.sidebar.selectbox(
    "Infill Pattern", options=["Cubic", "Grid", "Triangle"], index=0
)
extrusion_temp = st.sidebar.select_slider(
    "Extrusion Temperature (°C)", options=[190, 200, 210], value=200
)
bed_temp = st.sidebar.select_slider(
    "Build Plate Temperature (°C)", options=[50, 55, 60], value=55
)
print_speed = st.sidebar.select_slider(
    "Print Speed (mm/s)", options=[40, 60, 80], value=60
)
fan_speed = st.sidebar.select_slider(
    "Cooling Fan Speed (%)", options=[50, 75, 100], value=75
)

st.sidebar.header("4. Hybrid CNC Tolerance Target")
target_ra_limit = st.sidebar.number_input(
    "Target Surface Roughness Limit, Ra (µm)", min_value=5.0, max_value=35.0, value=15.0, step=0.5
)

# Build 1-row DataFrame for inference
input_row = pd.DataFrame(
    [
        {
            "Angle": float(angle),
            "Nozzle_Diameter": float(nozzle_diameter),
            "Layer_Height": float(layer_height),
            "Wall_Thickness": float(wall_thickness),
            "Top_Thickness": float(top_thickness),
            "Infill_Density": float(infill_density),
            "Infill_Pattern": int(PATTERN_MAP[infill_pattern_str]),
            "Extrusion_Temp": float(extrusion_temp),
            "Bed_Temp": float(bed_temp),
            "Print_Speed": float(print_speed),
            "Fan_Speed": float(fan_speed),
        }
    ]
)

predicted_ra, all_predictions, engine_mode = get_predictions(
    input_row, selected_algo, trained_models
)

# --- TABS ---
tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Prediction & Prescriptive Optimizer",
        "📊 Multi-Model Benchmark & Feature Importance",
        "📁 Taguchi L54 Dataset Explorer",
    ]
)

with tab1:
    col1, col2, col3 = st.columns([1.2, 1.2, 1.6])
    with col1:
        st.metric(
            label=f"Predicted Surface Roughness ({selected_algo})",
            value=f"{predicted_ra:.2f} µm",
            delta=f"{predicted_ra - target_ra_limit:+.2f} µm vs. CNC Limit",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            label="CAD Surface Angle (Fixed)",
            value=f"{angle:.0f}°",
            delta=f"Layer Height: {layer_height:.2f} mm",
            delta_color="off",
        )
    with col3:
        if predicted_ra <= target_ra_limit:
            st.success(
                f"**Within Tolerance (≤ {target_ra_limit:.1f} µm):** As-printed finish meets target; minimal or no hybrid CNC finishing allowance required."
            )
        else:
            st.warning(
                f"**Exceeds Target (> {target_ra_limit:.1f} µm):** Semi-finishing hybrid CNC pass or slicing parameter optimization recommended."
            )

    st.caption(f"Inference Mode: `{engine_mode}` | Dataset: Taguchi L54 (432 observations)")
    st.subheader("🚀 AI Prescriptive Process Optimizer (1,458 Combinations)")
    st.write(
        "Holds CAD **Surface Angle**, **Top Thickness**, **Infill Density**, and **Bed Temperature** constant, "
        "and evaluates **1,458 machine combinations** across 7 controllable slicing/thermal parameters to recommend "
        "the setup with minimum predicted $R_a$."
    )

    # Grid search across 2 * 3 * 3 * 3 * 3 * 3 * 3 = 1,458 combinations
    grid_rows = []
    for nz in [0.25, 0.40]:
        for lh in [0.10, 0.15, 0.20]:
            for wt in [0.75, 1.00, 1.25]:
                for pat in [0, 1, 2]:
                    for et in [190, 200, 210]:
                        for ps in [40, 60, 80]:
                            for fs in [50, 75, 100]:
                                grid_rows.append(
                                    (
                                        angle,
                                        nz,
                                        lh,
                                        wt,
                                        top_thickness,
                                        infill_density,
                                        pat,
                                        et,
                                        bed_temp,
                                        ps,
                                        fs,
                                    )
                                )

    search_df = pd.DataFrame(grid_rows, columns=FEATURE_COLS)
    active_estimator = trained_models[selected_algo]
    search_preds = active_estimator.predict(search_df)
    best_idx = int(np.argmin(search_preds))
    best_ra = round(float(search_preds[best_idx]), 3)
    best_row = search_df.iloc[best_idx]
    reduction_pct = (
        round(((predicted_ra - best_ra) / predicted_ra) * 100, 1)
        if predicted_ra > 0
        else 0.0
    )

    opt_c1, opt_c2, opt_c3 = st.columns(3)
    opt_c1.metric("Current Predicted Ra", f"{predicted_ra:.2f} µm")
    opt_c2.metric("Optimized Minimum Ra", f"{best_ra:.2f} µm")
    opt_c3.metric(
        "Roughness Reduction",
        f"-{max(reduction_pct, 0.0):.1f}%",
        delta=f"{best_ra - predicted_ra:+.2f} µm",
        delta_color="inverse",
    )

    comparison_table = pd.DataFrame(
        {
            "Process Parameter": [
                "Nozzle Diameter (mm)",
                "Layer Height (mm)",
                "Wall Thickness (mm)",
                "Infill Pattern",
                "Extrusion Temp (°C)",
                "Print Speed (mm/s)",
                "Fan Speed (%)",
            ],
            "Current Setting": [
                nozzle_diameter,
                layer_height,
                wall_thickness,
                infill_pattern_str,
                extrusion_temp,
                print_speed,
                fan_speed,
            ],
            "Recommended Optimal Setting": [
                best_row["Nozzle_Diameter"],
                best_row["Layer_Height"],
                best_row["Wall_Thickness"],
                INV_PATTERN_MAP[int(best_row["Infill_Pattern"])],
                int(best_row["Extrusion_Temp"]),
                int(best_row["Print_Speed"]),
                int(best_row["Fan_Speed"]),
            ],
        }
    )
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Live Multi-Model Predictions on Current Input")
    pred_comp_df = pd.DataFrame(
        [
            {"Model": m_name, "Predicted Ra (µm)": val}
            for m_name, val in all_predictions.items()
        ]
    )
    merged_df = pd.merge(pred_comp_df, metrics_df, on="Model", how="left")
    st.dataframe(merged_df, use_container_width=True, hide_index=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("**Test Set $R^2$ Score Comparison ($N_{\text{test}} = 87$)**")
        fig1, ax1 = plt.subplots(figsize=(6, 3.6))
        colors = [
            "#2E8B57" if r >= 0.90 else "#E67E22" for r in metrics_df["R2_Score"]
        ]
        ax1.barh(metrics_df["Model"], metrics_df["R2_Score"], color=colors, edgecolor="black")
        ax1.set_xlim(0, 1.08)
        ax1.set_xlabel("R² Score (Higher is Better)")
        ax1.grid(axis="x", linestyle="--", alpha=0.5)
        for idx, v in enumerate(metrics_df["R2_Score"]):
            ax1.text(v + 0.015, idx, f"{v:.4f}", va="center", fontsize=8.5)
        plt.tight_layout()
        st.pyplot(fig1)

    with c_right:
        st.markdown("**Feature Importance (Random Forest Regressor)**")
        rf_model = trained_models.get("Random Forest")
        if rf_model is not None and hasattr(rf_model, "feature_importances_"):
            fi_df = pd.DataFrame(
                {"Feature": FEATURE_COLS, "Importance": rf_model.feature_importances_}
            ).sort_values("Importance", ascending=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3.6))
            ax2.barh(fi_df["Feature"], fi_df["Importance"], color="#3B7EA1", edgecolor="black")
            ax2.set_xlabel("Relative Importance Score")
            ax2.grid(axis="x", linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig2)

with tab3:
    st.subheader("Mendeley FFF PLA Taguchi L54 Dataset (432 Experimental Rows)")
    if raw_df is not None:
        st.write(
            f"Showing summary statistics and experimental records from `am_data.xlsx` "
            f"({raw_df.shape[0]} rows × {raw_df.shape[1]} columns)."
        )
        st.dataframe(raw_df.describe().round(3), use_container_width=True)
        st.dataframe(raw_df.head(50), use_container_width=True, hide_index=True)
    else:
        st.info("Upload `am_data.xlsx` to view raw experimental rows and summary statistics.")
