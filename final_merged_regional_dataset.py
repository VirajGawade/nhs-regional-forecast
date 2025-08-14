import pandas as pd
import os

base_dir = r"C:\Users\viraj\Downloads\nhs regional scripts\cleaned datasets"

# Mapping from detailed NHS region names to unified simpler names used by ECD datasets
region_map = {
    "NHS ENGLAND EAST OF ENGLAND": "East of England",
    "NHS ENGLAND LONDON": "London",
    "NHS ENGLAND MIDLANDS": "Midlands",
    "NHS ENGLAND NORTH EAST AND YORKSHIRE": "North East and Yorkshire",
    "NHS ENGLAND NORTH WEST": "North West",
    "NHS ENGLAND SOUTH EAST": "South East",
    "NHS ENGLAND SOUTH WEST": "South West"
}

def unify_region(region_name):
    for prefix, unified in region_map.items():
        if region_name.startswith(prefix):
            return unified
    return region_name  # fallback: return original if no match

# Load base A&E dataset
merged = pd.read_csv(os.path.join(base_dir, "multivariate_monthly_regional_ae.csv"))
merged["Month"] = pd.to_datetime(merged["Month"]).dt.to_period("M").dt.to_timestamp()
merged["Region_unified"] = merged["Region"].apply(unify_region)

# Load and process ECDS Performance
ecd_perf = pd.read_csv(os.path.join(base_dir, "ecd_regional_performance.csv"))
ecd_perf["Month"] = pd.to_datetime(ecd_perf["Month"]).dt.to_period("M").dt.to_timestamp()
ecd_perf["Region_unified"] = ecd_perf["Region"].apply(unify_region)
ecd_perf.drop(columns=["Region"], inplace=True)  # drop original to avoid merge conflicts

# Load and process ECDS Distributions
ecd_dist = pd.read_csv(os.path.join(base_dir, "ecd_regional_distributions_pivoted.csv"))
ecd_dist["Month"] = pd.to_datetime(ecd_dist["Month"]).dt.to_period("M").dt.to_timestamp()
ecd_dist["Region_unified"] = ecd_dist["Region"].apply(unify_region)
ecd_dist.drop(columns=["Region"], inplace=True)

# Load and process NEL
nel = pd.read_csv(os.path.join(base_dir, "multivariate_monthly_regional_with_NEL.csv"))
nel["Month"] = pd.to_datetime(nel["Month"]).dt.to_period("M").dt.to_timestamp()
nel["Region_unified"] = nel["Region"].apply(unify_region)
nel.drop(columns=["Region"], inplace=True)

# Load and process Nomis
nomis = pd.read_csv(os.path.join(base_dir, "nomis_regional_cleaned.csv"))
nomis = nomis[nomis["Year"] >= 2020].copy()
nomis.rename(columns={"Year": "Month"}, inplace=True)
nomis["Month"] = pd.to_datetime(nomis["Month"], format="%Y").dt.to_period("M").dt.to_timestamp()
nomis["Region_unified"] = nomis["Region"].apply(unify_region)
nomis.drop(columns=["Region"], inplace=True)

# Load and process Fingertips (national data, no region)
fingertips = pd.read_csv(os.path.join(base_dir, "fingertips_cleaned.csv"))
fingertips.rename(columns={"Year": "Month"}, inplace=True)
fingertips["Month"] = pd.to_datetime(fingertips["Month"], format="%Y").dt.to_period("M").dt.to_timestamp()

# Load and process MetOffice (national data, no region)
weather = pd.read_csv(os.path.join(base_dir, "MetOffice_cleaned_dataset.csv"))
weather["Month"] = pd.to_datetime(weather[["Year", "Month"]].astype(str).agg("-".join, axis=1), format="%Y-%m")
weather.drop(columns=["Year"], inplace=True)

# Load and process UKHSA (national data, no region)
ukhsa = pd.read_csv(os.path.join(base_dir, "ukhsa_cleaned.csv"))
ukhsa.rename(columns={"date": "Month"}, inplace=True)
ukhsa["Month"] = pd.to_datetime(ukhsa["Month"]).dt.to_period("M").dt.to_timestamp()

# Load and process Google Trends (national data, no region)
google = pd.read_csv(os.path.join(base_dir, "google_trends_cleaned.csv"))
google.rename(columns={"Date": "Month"}, inplace=True)
google["Month"] = pd.to_datetime(google["Month"]).dt.to_period("M").dt.to_timestamp()

# Merge datasets with unified region + Month keys
region_dfs = [ecd_perf, ecd_dist, nel, nomis]
for df in region_dfs:
    merged = pd.merge(merged, df, on=["Month", "Region_unified"], how="left")

# Merge datasets on Month only (national datasets)
for df in [weather, ukhsa, google, fingertips]:
    merged = pd.merge(merged, df, on="Month", how="left")

# Missing value imputation
#  Fill flu, covid, syndrome signals by calendar month median
for col in merged.columns:
    if any(sig in col.lower() for sig in ["flu", "covid", "syndrome"]):
        merged[col] = merged.groupby(merged["Month"].dt.month)[col].transform(lambda x: x.fillna(x.median()))

#  Fill lag/rolling features by forward fill then backward fill
for col in merged.columns:
    if any(kw in col.lower() for kw in ["lag", "rolling", "roll"]):
        merged[col] = merged[col].ffill().bfill()

#  Fill environmental/weather features by calendar month median
for col in merged.columns:
    if any(env in col.lower() for env in ["temp", "rain", "humidity", "wind", "weather"]):
        if merged[col].isna().sum() > 0:
            merged[col] = merged.groupby(merged["Month"].dt.month)[col].transform(lambda x: x.fillna(x.median()))

#  For other numeric columns, cascade fill forward, backward then median
for col in merged.columns:
    if pd.api.types.is_numeric_dtype(merged[col]) and merged[col].isna().sum() > 0:
        merged[col] = merged[col].ffill().bfill().fillna(merged[col].median())

#  Clean up _x/_y duplicate columns
cols = merged.columns.tolist()
base_names = set()
for col in cols:
    if col.endswith('_x'):
        base = col[:-2]
        if f"{base}_y" in cols:
            base_names.add(base)

# Collect all new cleaned columns in a dictionary first to avoid fragmentation
new_cols = {}
for base in base_names:
    col_x = f"{base}_x"
    col_y = f"{base}_y"

    x_na = merged[col_x].isna().sum()
    y_na = merged[col_y].isna().sum()

    if x_na <= y_na:
        new_cols[base] = merged[col_x]
    else:
        new_cols[base] = merged[col_y]

# Add all cleaned columns at once
merged = pd.concat([merged, pd.DataFrame(new_cols)], axis=1)

# Drop original _x and _y columns
cols_to_drop = [col for base in base_names for col in (f"{base}_x", f"{base}_y")]
merged.drop(columns=cols_to_drop, inplace=True)

# Final sort and export
merged = merged.sort_values(["Region_unified", "Month"], ascending=[True, True]).reset_index(drop=True)

# Load national column order (without Region columns)
national_order = pd.read_csv("final_merged_dataset.csv", nrows=1).columns.tolist()
regional_columns = merged.columns.tolist()

# Remove Region and Region_unified if already in national_order for safety
base_cols = [col for col in national_order if col not in ["Region", "Region_unified"]]

# Insert Region and Region_unified after Month
if "Month" in base_cols:
    month_index = base_cols.index("Month")
    base_cols.insert(month_index + 1, "Region")
    base_cols.insert(month_index + 2, "Region_unified")

# Handle any extra regional columns not in national dataset
extra_cols = [col for col in regional_columns if col not in base_cols]
final_order = base_cols + extra_cols

# Apply column order
merged = merged[final_order]
merged["Type"] = merged["Type"].fillna("T1")
# REMOVE exact duplicate rows (keep only first occurrence)
merged = merged.drop_duplicates(keep='first')
merged.to_csv("final_merged_regional_dataset.csv", index=False)

print(" Output saved to final_merged_regional_dataset.csv")
