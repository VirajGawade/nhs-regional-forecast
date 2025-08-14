import os
import re
import pandas as pd
from glob import glob
from difflib import get_close_matches

# Config 
MULTIVARIATE_PATH = "multivariate_monthly_regional_ae.csv"
NEL_FOLDER = r"C:\Users\viraj\Downloads\nhs project dataset\NEL Growth Rate Files"
OUTPUT_PATH = "multivariate_monthly_regional_with_NEL.csv"

# Load multivariate regional dataset 
df_multi = pd.read_csv(MULTIVARIATE_PATH)
df_multi['Month'] = pd.to_datetime(df_multi['Month']).dt.to_period("M").astype(str)

# Extract NEL growth data dynamically from raw Excel files
def find_column(df_cols, possible_names):
    df_cols_lower = [c.lower() for c in df_cols]
    for name in possible_names:
        matches = get_close_matches(name.lower(), df_cols_lower, n=1, cutoff=0.7)
        if matches:
            return df_cols[df_cols_lower.index(matches[0])]
    return None

all_growth_rows = []

for file_path in glob(os.path.join(NEL_FOLDER, "*NEL-YTD-Growth-rates*.xls*")):
    filename = os.path.basename(file_path)
    try:
        # Extract month and year from filename
        match = re.search(r"([A-Za-z]+)[ -]+(\d{4})", filename)
        if not match:
            print(f" Could not parse date from filename: {filename}")
            continue
        month_str = pd.to_datetime(" ".join(match.groups()), format="%B %Y").strftime("%Y-%m")

        # Read appropriate sheet
        try:
            df = pd.read_excel(file_path, sheet_name="Emergency", header=15)
        except:
            df = pd.read_excel(file_path, sheet_name=0, header=15)

        df.columns = [str(c).strip() for c in df.columns]
        region_col = find_column(df.columns, ["Region", "Name"])
        growth_col = find_column(df.columns, ["Year to Date Emergency Total Growth"])

        if region_col is None or growth_col is None:
            print(f" Missing Region or Growth columns in: {filename}")
            continue

        temp_df = df[[region_col, growth_col]].copy()
        temp_df.rename(columns={region_col: "Region", growth_col: "NEL_YTD_Growth"}, inplace=True)
        temp_df["Region"] = temp_df["Region"].astype(str).str.strip().str.upper()

        def to_float(x):
            if isinstance(x, str):
                x = x.replace("%", "").strip()
                return None if x in ["", "*"] else float(x) / 100
            return float(x) if pd.notna(x) else None

        temp_df["NEL_YTD_Growth"] = temp_df["NEL_YTD_Growth"].apply(to_float)
        temp_df["Month"] = month_str

        all_growth_rows.append(temp_df)

    except Exception as e:
        print(f" Error processing {filename}: {e}")

# Combine growth data
if not all_growth_rows:
    raise ValueError(" No NEL data was loaded. Please check the Excel files in the folder.")

df_nel = pd.concat(all_growth_rows, ignore_index=True)
df_nel = df_nel.groupby(["Month", "Region"], as_index=False)["NEL_YTD_Growth"].mean()

# Prepare for merge 
def simplify_region(region):
    if pd.isna(region):
        return None
    return region.split("(")[0].strip().upper()

df_multi['Region_clean'] = df_multi['Region'].apply(simplify_region)
df_nel['Region_clean'] = df_nel['Region'].str.strip().str.upper()
df_nel['Month'] = pd.to_datetime(df_nel['Month']).dt.to_period("M").astype(str)

# Merge on Month and Region_clean 
df_merged = pd.merge(
    df_multi,
    df_nel[['Month', 'Region_clean', 'NEL_YTD_Growth']],
    how='left',
    left_on=['Month', 'Region_clean'],
    right_on=['Month', 'Region_clean']
)

df_merged.drop(columns=['Region_clean'], inplace=True)

# Save result 
df_merged.to_csv(OUTPUT_PATH, index=False)
print(f" Final merged file saved to {OUTPUT_PATH}")
