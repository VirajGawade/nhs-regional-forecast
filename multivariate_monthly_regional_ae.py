import os
import pandas as pd
import re
from glob import glob
from difflib import get_close_matches

DATA_FOLDER = r"C:\Users\viraj\Downloads\nhs project dataset\monthly csv dataset"
OUTPUT_FILE = r"C:\Users\viraj\Downloads\nhs regional scripts\multivariate_monthly_regional_ae.csv"

# Define canonical column names you want to extract
COLUMN_MAP = {
    "Type1_Attendances": ["A&E attendances Type 1", "Number of A&E attendances Type 1", "AE attendances Type 1", "A&E attendance Type 1"],
    "Type2_Attendances": ["A&E attendances Type 2", "Number of A&E attendances Type 2", "AE attendances Type 2", "A&E attendance Type 2"],
    "Other_Attendances": ["A&E attendances Other A&E Department", "Number of A&E attendances Other A&E Department", "AE attendances Other Department"],
    "Over_4hr_Type1": ["Attendances over 4hrs Type 1", "Number of attendances over 4hrs Type 1", "Attendances over 4 hours Type 1"],
    "Over_4hr_Type2": ["Attendances over 4hrs Type 2", "Number of attendances over 4hrs Type 2", "Attendances over 4 hours Type 2"],
    "Over_4hr_Other": ["Attendances over 4hrs Other Department", "Number of attendances over 4hrs Other A&E Department", "Attendances over 4 hours Other Department"],
    "Over_12hr": ["Patients who have waited 12+ hrs from DTA to admission"],
    "Admissions_Type1": ["Emergency admissions via A&E - Type 1"],
    "Admissions_Type2": ["Emergency admissions via A&E - Type 2"],
    "Admissions_Other": ["Emergency admissions via A&E - Other A&E department"]
}

def find_column(df_cols, possible_names):
    """Find closest matching column in df_cols from possible_names."""
    df_cols_lower = [c.lower() for c in df_cols]
    for name in possible_names:
        matches = get_close_matches(name.lower(), df_cols_lower, n=1, cutoff=0.7)
        if matches:
            # Return original case-sensitive column name
            return df_cols[df_cols_lower.index(matches[0])]
    return None

feature_rows = []

for file_path in glob(os.path.join(DATA_FOLDER, "*.csv")):
    filename = os.path.basename(file_path)

    match = re.search(r"(?:Monthly-AE-)?([A-Za-z]+)-(\d{4})", filename)
    if not match:
        if filename.lower() == "multivariate_monthly_regional_ae.csv":
            continue
        print(f" Could not parse date from filename: {filename}")
        continue

    month_name, year = match.groups()
    try:
        month_str = pd.to_datetime(f"{month_name} {year}", format="%B %Y").strftime("%Y-%m")
    except Exception:
        print(f" Invalid month format in filename: {filename}")
        continue

    try:
        df = pd.read_csv(file_path)

        # Identify region column
        region_col = None
        for candidate in ["Region", "Parent Org", "Org name"]:
            if candidate in df.columns:
                region_col = candidate
                break
        if region_col is None:
            raise KeyError("No 'Region', 'Parent Org', or 'Org name' column found")

        # Exclude total rows robustly 
        df = df[~df[region_col].astype(str).str.strip().str.upper().eq("TOTAL")]

        # Find all columns we want using fuzzy matching
        cols_found = {}
        for key, possible_names in COLUMN_MAP.items():
            col_name = find_column(df.columns, possible_names)
            if col_name is not None:
                cols_found[key] = col_name
            else:
                cols_found[key] = None  # Will handle missing columns

        grouped = df.groupby(region_col)

        for region_name, group in grouped:
            def safe_sum(col_key):
                col = cols_found.get(col_key)
                if col and col in group.columns:
                    return group[col].sum()
                return 0

            type1 = safe_sum("Type1_Attendances")
            type2 = safe_sum("Type2_Attendances")
            other = safe_sum("Other_Attendances")
            over_4hr = safe_sum("Over_4hr_Type1") + safe_sum("Over_4hr_Type2") + safe_sum("Over_4hr_Other")
            over_12hr = safe_sum("Over_12hr")
            admissions = safe_sum("Admissions_Type1") + safe_sum("Admissions_Type2") + safe_sum("Admissions_Other")

            feature_rows.append({
                "Month": month_str,
                "Region": region_name,
                "Total_Attendances": type1 + type2 + other,
                "Type1_Attendances": type1,
                "Type2_Attendances": type2,
                "Other_Attendances": other,
                "Over_4hr_Waits": over_4hr,
                "Over_12hr_Waits": over_12hr,
                "Emergency_Admissions": admissions
            })

    except Exception as e:
        print(f" Error processing {filename}: {e}")

final_df = pd.DataFrame(feature_rows).sort_values(["Month", "Region"])
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Multivariate regional dataset saved to:\n{OUTPUT_FILE}")
print(final_df.head())
