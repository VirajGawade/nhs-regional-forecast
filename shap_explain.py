import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Config
MONTHLY_DATA = "regional_engineered_features.csv"
WEEKLY_DATA  = "regional_engineered_features_weekly.csv"

ML_MONTHLY_MODELS  = "regional_ml_monthly_models"
ML_MONTHLY_SCALERS = "regional_ml_monthly_scalers"

ML_WEEKLY_MODELS   = "regional_ml_weekly_models"
ML_WEEKLY_SCALERS  = "regional_ml_weekly_scalers"

TARGET     = "Total_Attendances"
REGION_COL = "Region_unified"
OUT_DIR    = "shap_outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# SHAP compatibility
import shap

def make_explainer(model):
    """
    Prefer unified API if available; otherwise fall back to TreeExplainer
    (works for XGBoost / RandomForest on older SHAP versions).
    """
    try:
        return shap.Explainer(model)  # SHAP >= 0.36
    except AttributeError:
        return shap.TreeExplainer(model)  

def plot_shap_bar(shap_values, max_display=15, out_path=None):
    """
    SHAP plotting API changed across versions; keep robust.
    """
    try:
        plt.figure()
        shap.plots.bar(shap_values, show=False, max_display=max_display)
    except Exception:
        # Older SHAP fallback
        plt.figure()
        shap.summary_plot(shap_values, plot_type="bar", show=False, max_display=max_display)
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def plot_shap_waterfall(shap_values, out_path=None, max_display=15):
    """
    Waterfall with dynamic left-margin adjustment:
    
    """
    try:
        from matplotlib import transforms
        import re

        fig = plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_values[0], show=False, max_display=max_display)

        ax = plt.gca()
        fig.canvas.draw()  # ensure artists exist

        # collect numeric labels and their y positions
        num_re = re.compile(r"^[\+\-−]?\d+(\.\d+)?(e[\+\-]?\d+)?$")
        labels = []
        neg_text_objs = []
        for txt in list(ax.texts):
            s = (txt.get_text() or "").strip()
            if num_re.match(s):
                labels.append((s, txt.get_position()[1]))
                if s.startswith("-") or s.startswith("−"):
                    neg_text_objs.append(txt)
                txt.set_visible(False)  # hide originals

        # dynamic left margin based on widest blue label 
        new_left = 0.50
        if neg_text_objs:
            renderer = fig.canvas.get_renderer()
            max_w_px = max(t.get_window_extent(renderer=renderer).width for t in neg_text_objs)
            ax_bb_px = ax.get_window_extent(renderer=renderer).width
            extra_left_frac = (max_w_px / ax_bb_px) + 0.08  # more padding
            new_left = min(0.85, max(0.50, 0.50 + extra_left_frac))


        # redraw all numeric labels on the right side
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for s, y in labels:
            ax.text(0.99, y, s, transform=trans, ha="right", va="center")

        # spacing + crisp save
        plt.tight_layout()
        plt.subplots_adjust(left=new_left, right=0.98)
        if out_path:
            plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    except Exception:
        pass

# utilities
def numeric_feature_cols(df, drop_cols):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]

def ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[cols]


# monthly
def explain_monthly(region: str, year: int, month: int, model_name: str):
    """
    Explains a single monthly prediction for a region using the saved ML model.
    - Validation: if target month exists -> explain that row.
    - Forecast: mean of last 12 months' features.
    Saves SHAP bar (+waterfall if supported) to shap_outputs/.
    """
    df = pd.read_csv(MONTHLY_DATA, parse_dates=["Month"]).sort_values("Month")
    df = df.dropna(subset=[TARGET])
    region_df = df[df[REGION_COL] == region].copy().sort_values("Month").reset_index(drop=True)

    non_features = [TARGET, REGION_COL, "Month", "Region", "Type"]
    feature_cols = numeric_feature_cols(region_df, non_features)

    # Load scaler & model
    scaler_path = os.path.join(ML_MONTHLY_SCALERS, f"{region.replace(' ','_')}_scaler.save")
    model_path  = os.path.join(ML_MONTHLY_MODELS,  f"{region.replace(' ','_')}_{model_name}_monthly.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    target_date = pd.Timestamp(year=year, month=month, day=1)
    last_month  = region_df["Month"].max()

    if target_date <= last_month and target_date in set(region_df["Month"]):
        X = ensure_cols(region_df[region_df["Month"] == target_date], feature_cols)
        actual = float(region_df.loc[region_df["Month"] == target_date, TARGET].values[0])
        mode = "validation"
    else:
        hist = region_df[region_df["Month"] < target_date].tail(12)
        if len(hist) < 12:
            raise ValueError("Not enough history for 12-month mean.")
        X = pd.DataFrame([hist[feature_cols].mean()], columns=feature_cols)
        actual = None
        mode = "forecast"

    # keep feature names after scaling
    Xs = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols
    )

    # SHAP explain
    explainer   = make_explainer(model)
    shap_values = explainer(Xs)
    pred        = float(model.predict(Xs)[0])

    base = f"monthly_{region.replace(' ','_')}_{model_name}_{target_date.date()}_{mode}"
    bar_png = os.path.join(OUT_DIR, base + "_bar.png")
    wtf_png = os.path.join(OUT_DIR, base + "_waterfall.png")

    plot_shap_bar(shap_values, max_display=15, out_path=bar_png)
    plot_shap_waterfall(shap_values, out_path=wtf_png, max_display=15)

    return {
        "granularity": "monthly",
        "region": region,
        "model": model_name,
        "date": str(target_date.date()),
        "mode": mode,
        "prediction": pred,
        "actual": actual,
        "bar_png": bar_png,
        "waterfall_png": wtf_png
    }


# weekly
def explain_weekly(region: str, year: int, month: int, day: int, model_name: str):
    """
    Explains a single weekly prediction for a region using the saved ML model.
    - Validation: if week_end exists -> mean of last 12 weeks' features prior to week_end.
    - Forecast: mean of last 12 weeks' features from the tail.
    Saves SHAP bar (+waterfall if supported) to shap_outputs/.
    """
    df = pd.read_csv(WEEKLY_DATA, parse_dates=["Week"]).sort_values("Week")
    region_df = df[df[REGION_COL] == region].copy().sort_values("Week").reset_index(drop=True)

    non_features = [TARGET, REGION_COL, "Week", "Region", "Type"]
    feature_cols = numeric_feature_cols(region_df, non_features)

    # Determine ISO week Monday/Sunday
    input_date = pd.Timestamp(f"{year:04d}-{month:02d}-{day:02d}")
    week_start = input_date - pd.Timedelta(days=input_date.weekday())  # Monday
    week_end   = week_start + pd.Timedelta(days=6)                     # Sunday

    # Load scaler & model
    scaler_path = os.path.join(ML_WEEKLY_SCALERS, f"weekly_scaler_{region}.save")
    model_path  = os.path.join(ML_WEEKLY_MODELS,  f"{region.replace(' ','_')}_{model_name}_weekly.pkl")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    scaler = joblib.load(scaler_path)
    model  = joblib.load(model_path)

    # Build input per ML weekly logic
    if week_end in set(region_df["Week"]):
        window = region_df[region_df["Week"] < week_end].tail(12)
        if len(window) < 12:
            raise ValueError("Not enough history for 12-week validation window.")
        actual = float(region_df.loc[region_df["Week"] == week_end, TARGET].values[0])
        mode = "validation"
    else:
        window = region_df.tail(12)
        if len(window) < 12:
            raise ValueError("Not enough history for 12-week forecast window.")
        actual = None
        mode = "forecast"

    X = pd.DataFrame([window[feature_cols].mean()], columns=feature_cols)

    # Align to scaler expected columns
    expected = list(getattr(scaler, "feature_names_in_", feature_cols))
    if TARGET in expected:
        expected.remove(TARGET)
    # Ensure all exist
    X = ensure_cols(X, expected)

    # keep feature names after scaling
    Xs = pd.DataFrame(
        scaler.transform(X),
        columns=expected
    )

    # SHAP explain
    explainer   = make_explainer(model)
    shap_values = explainer(Xs)
    pred        = float(model.predict(Xs)[0])

    iso_year, iso_week, _ = week_end.isocalendar()
    base = f"weekly_{region.replace(' ','_')}_{model_name}_{iso_year}-W{str(iso_week).zfill(2)}_{mode}"
    bar_png = os.path.join(OUT_DIR, base + "_bar.png")
    wtf_png = os.path.join(OUT_DIR, base + "_waterfall.png")

    plot_shap_bar(shap_values, max_display=15, out_path=bar_png)
    plot_shap_waterfall(shap_values, out_path=wtf_png, max_display=15)

    return {
        "granularity": "weekly",
        "region": region,
        "model": model_name,
        "week_end": str(week_end.date()),
        "mode": mode,
        "prediction": pred,
        "actual": actual,
        "bar_png": bar_png,
        "waterfall_png": wtf_png
    }


# CLI 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SHAP explanations for NHS A&E ML models (monthly/weekly).")
    p.add_argument("--granularity", choices=["monthly", "weekly"], required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--model", choices=["RandomForest", "XGBoost"], required=True)
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--day", type=int, help="Only required for weekly granularity")
    args = p.parse_args()

    if args.granularity == "monthly":
        out = explain_monthly(args.region, args.year, args.month, args.model)
    else:
        if args.day is None:
            raise SystemExit("--day is required for weekly")
        out = explain_weekly(args.region, args.year, args.month, args.day, args.model)

    print("[INFO] SHAP explanation complete:")
    for k, v in out.items():
        print(f"  {k}: {v}")
