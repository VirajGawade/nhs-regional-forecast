import streamlit as st
import subprocess
import sys
from pathlib import Path
import glob
import subprocess
import time

# ===================== CONFIG =====================
BASE_DIR = Path(__file__).resolve().parent

REGIONS = [
    "East of England",
    "London",
    "Midlands",
    "North East and Yorkshire",
    "North West",
    "South East",
    "South West"
]

GRANULARITIES = ["Monthly", "Weekly", "Daily"]

SCRIPT_MAP = {
    "Monthly": {
        "Hybrid": "Hybrid_monthly_pred.py",
        "ML": "Regional_ML_1month_pred.py",
        "LSTM": "Regional_LSTM_1month_pred.py"
    },
    "Weekly": {
        "Hybrid": "Hybrid_weekly_pred.py",
        "ML": "Regional_ML_1week_pred.py",
        "LSTM": "Regional_LSTM_1week_pred.py"
    },
    "Daily": {
        "Hybrid": "Hybrid_7day_daily.py",
        "ML": "Regional_ML_7day_daily_pred.py",
        "LSTM": "Regional_LSTM_7day_daily_pred.py"
    }
}

# ===================== HELPERS =====================
def run_script(script_name, year, month, day=None):
    """Run a standalone script with given parameters and return its output."""
    script_path = BASE_DIR / script_name
    if not script_path.exists():
        return f"[ERROR] Script not found: {script_path}"

    cmd = [sys.executable, str(script_path), "--year", str(year), "--month", str(month)]
    if day is not None:
        cmd.extend(["--day", str(day)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stdout + "\n" + e.stderr

def filter_by_region(output_text, region):
    """Filter CLI output to only show lines for the selected region."""
    if not region or "Processing region" not in output_text:
        return output_text

    filtered_lines = []
    capture = False
    for line in output_text.splitlines():
        # Start capturing when chosen region appears
        if f"Processing region: {region}" in line:
            capture = True
        # Stop capturing when we see another 'Processing region:' line
        elif line.startswith("[INFO] Processing region:") and capture:
            break
        if capture:
            filtered_lines.append(line)

    return "\n".join(filtered_lines) if filtered_lines else "[No matching region found]"



# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="NHS Regional Forecast", layout="wide")

st.title("📊 NHS Regional Attendances Forecast")
st.caption("Forecasts match the logic of standalone scripts exactly.")

granularity = st.selectbox("Select Granularity", GRANULARITIES)
region = st.selectbox("Select Region", ["All"] + REGIONS)
year = st.number_input("Year", min_value=2000, max_value=2100, value=2025)
month = st.number_input("Month", min_value=1, max_value=12, value=5)

day = None
if granularity in ["Weekly", "Daily"]:
    day = st.number_input("Day", min_value=1, max_value=31, value=1)

tabs = st.tabs(["Hybrid (Default)", "ML", "LSTM", "SHAP Explainability (ML Only)"])

# --- Hybrid tab ---
with tabs[0]:
    script = SCRIPT_MAP[granularity]["Hybrid"]
    output = run_script(script, year, month, day)
    if region != "All":
        output = filter_by_region(output, region)
    st.subheader(f"{granularity} - Hybrid Prediction")
    st.code(output)

# --- ML tab ---
with tabs[1]:
    script = SCRIPT_MAP[granularity]["ML"]
    output = run_script(script, year, month, day)
    if region != "All":
        output = filter_by_region(output, region)
    st.subheader(f"{granularity} - ML Prediction")
    st.code(output)

# --- LSTM tab ---
with tabs[2]:
    script = SCRIPT_MAP[granularity]["LSTM"]
    output = run_script(script, year, month, day)
    if region != "All":
        output = filter_by_region(output, region)
    st.subheader(f"{granularity} - LSTM Prediction")
    st.code(output)

# --- SHAP tab ---
with tabs[3]:
    st.header("SHAP Explainability (ML Models Only)")

    granularity_shap = st.selectbox("Granularity", ["monthly", "weekly"])
    region_shap = st.selectbox("Region", REGIONS)
    model_shap = st.selectbox("Model", ["RandomForest", "XGBoost"])

    year_shap = st.number_input("Year (SHAP)", min_value=2000, max_value=2100, value=2025)
    month_shap = st.number_input("Month (SHAP)", min_value=1, max_value=12, value=1)
    day_shap = None
    if granularity_shap != "monthly":
        day_shap = st.number_input("Day (SHAP)", min_value=1, max_value=31, value=1)

    if st.button("Run SHAP Explainability"):
        shap_dir = Path("shap_outputs")
        shap_dir.mkdir(exist_ok=True)  # Ensure folder exists

        # 1. Remove old SHAP plots from shap_outputs/
        for old_png in glob.glob(str(shap_dir / "*.png")):
            Path(old_png).unlink(missing_ok=True)

        # 2. Build the command
        cmd = [
            sys.executable, "shap_explain.py",
            "--granularity", granularity_shap,
            "--region", region_shap,
            "--model", model_shap,
            "--year", str(year_shap),
            "--month", str(month_shap)
        ]
        if day_shap is not None:
            cmd += ["--day", str(day_shap)]

        # 3. Run script
        st.write("Running SHAP explainability...")
        subprocess.run(cmd)

        time.sleep(1)  # give time for PNGs to save

        # 4. Load last 2 PNGs created from shap_outputs folder
        png_files = sorted(
            glob.glob(str(shap_dir / "*.png")),
            key=lambda x: Path(x).stat().st_mtime,
            reverse=True
        )[:2]

        if png_files:
            st.success("SHAP plots generated:")

            # Display side-by-side with smaller width
            col1, col2 = st.columns(2)
            if len(png_files) >= 1:
                col1.image(png_files[1], width=550)  # Bar plot
            if len(png_files) >= 2:
                col2.image(png_files[0], width=550)  # Waterfall plot
        else:
            st.error("No SHAP PNG files found.")


