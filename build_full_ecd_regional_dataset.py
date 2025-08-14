import os
import re
import glob
import pandas as pd
from datetime import datetime

# config
SUPP_DIR = r"C:\Users\viraj\Downloads\nhs project dataset\Supplementary ECDS Analyses"

def parse_month_from_filename(fn):
    m1 = re.search(r'Analysis-([A-Za-z]+-\d{4})', fn)
    m2 = re.search(r'Analysis-([A-Za-z]+)-(\d{2})', fn)
    if m1:
        return datetime.strptime(m1.group(1), '%B-%Y').strftime('%Y-%m')
    elif m2:
        return datetime.strptime(f"{m2.group(1)}-20{m2.group(2)}", '%B-%Y').strftime('%Y-%m')
    raise ValueError(f"Cannot parse month-year from {fn}")

def find_header_row(raw, keyword):
    mask = raw.apply(lambda r: r.astype(str).str.contains(keyword, case=False, na=False).any(), axis=1)
    if not mask.any():
        raise RuntimeError(f"Header row with '{keyword}' not found")
    return mask.idxmax()

def find_sheet_name(sheets, base):
    for s in sheets:
        if base in s and ('T1' in s or 'Type 1' in s):
            return s
    raise ValueError(f"Worksheet named '{base} - T1' or variant not found")

def extract_summary(path, base_sheet):
    sheets = pd.ExcelFile(path).sheet_names
    try:
        sheet = find_sheet_name(sheets, base_sheet)
    except ValueError as e:
        raise RuntimeError(e)

    raw = pd.read_excel(path, sheet_name=sheet, header=None, dtype=str)
    hr = find_header_row(raw, 'Total Attendances')
    df = pd.read_excel(path, sheet_name=sheet, header=hr, dtype=str)

    df = df[df['Region'].notna() & (df['Region'].str.strip() != '')]
    region_rows = []
    for _, row in df.iterrows():
        entry = {
            'Region': row['Region'].strip(),
            'Total Attendances': pd.to_numeric(row.get('Total Attendances', None), errors='coerce'),
            'Admitted Attendances': pd.to_numeric(row.get('Admitted Attendances', None), errors='coerce'),
        }
        for col in row.index:
            if col.strip().lower() == 'a&e attendances >12hrs from arrival':
                entry['Delays >12hrs (count)'] = pd.to_numeric(row[col], errors='coerce')
            if col.strip() == '12hr %':
                entry['12hr %'] = pd.to_numeric(row[col], errors='coerce')
        region_rows.append(entry)

    return region_rows

def extract_distribution(path, metric_prefix):
    sheet_guess = f"{metric_prefix.title().replace('_', ' ')} - T1"
    try:
        raw = pd.read_excel(path, sheet_name=sheet_guess, header=None, dtype=str)
    except Exception:
        return None

    try:
        hr = find_header_row(raw, 'Region')
    except RuntimeError:
        return None
    df = pd.read_excel(path, sheet_name=sheet_guess, header=hr, dtype=str)

    df = df[df['Region'].notna() & (df['Region'].str.strip() != '')]

    region_rows = []
    skip_cols = {'Region', 'Org Code', 'Org Name'}

    for _, row in df.iterrows():
        region = row['Region'].strip()
        out = {'Region': region}
        for col in df.columns:
            if col in skip_cols:
                continue
            suffix = '_delayed' if col.endswith('.1') else ''
            base_col = col[:-2] if suffix else col
            col_clean = re.sub(r'[^0-9A-Za-z]+', '_', base_col.strip()).strip('_').lower()
            key = f"{metric_prefix}_{col_clean}{suffix}"
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                out[key] = val
        if len(out) > 1:
            region_rows.append(out)

    return region_rows if region_rows else None

def main():
    xls_pattern = os.path.join(SUPP_DIR, 'Supplementary-ECDS-Analysis-*-Final*.xlsx')
    files = sorted(glob.glob(xls_pattern))
    perf_rows, dist_rows = [], []

    CLEAN_DIR = r"C:\Users\viraj\Downloads\nhs regional scripts"
    perf_out = os.path.join(CLEAN_DIR, 'ecd_regional_performance.csv')
    dist_out = os.path.join(CLEAN_DIR, 'ecd_regional_distributions.csv')
    for f in (perf_out, dist_out):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    for fp in files:
        fn = os.path.basename(fp)
        try:
            month = parse_month_from_filename(fn)
        except Exception as e:
            print(f" Skipped {fn} → {e}")
            continue

        try:
            perf_list = extract_summary(fp, "System & Provider Summary")
            for row in perf_list:
                perf_rows.append({'Month': month, 'Type': 'T1', **row})
        except Exception as e:
            print(f" Skipped summary: {fn} (T1) → {e}")

        for metric in ('age', 'gender', 'ethnicity', 'chief_complaint'):
            try:
                dist_list = extract_distribution(fp, metric)
                if dist_list:
                    for row in dist_list:
                        dist_rows.append({'Month': month, 'Type': 'T1', 'Metric': metric.title().replace('_', ' '), **row})
            except Exception as e:
                print(f" Skipped dist: {fn} ({metric}) → {e}")

    perf_df = pd.DataFrame(perf_rows)
    dist_df = pd.DataFrame(dist_rows)
    dist_df = dist_df.dropna(axis=0, how='all', subset=dist_df.columns.difference(['Month', 'Type', 'Metric', 'Region']))

    # Drop Region rows starting with footnotes like "1. ", "2. " or "Notes:"
    perf_df = perf_df[~perf_df['Region'].astype(str).str.match(r'^(Notes:|\d+\.\s)')]

    # Remove rows where Region is '-' (national total rows)
    perf_df = perf_df[perf_df['Region'].astype(str).str.strip() != '-']
    dist_df = dist_df[dist_df['Region'].astype(str).str.strip() != '-']  # Remove from distributions too

    # Aggregate provider rows per region per month
    numeric_cols = [col for col in perf_df.columns if col not in ['Month', 'Region', 'Type']]
    perf_df[numeric_cols] = perf_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    perf_df = perf_df.groupby(['Month', 'Region', 'Type'], as_index=False)[numeric_cols].sum()
    perf_df = perf_df.sort_values(by=['Month', 'Region'])

    dist_df = dist_df.sort_values(by=['Month', 'Region'])
    perf_df.to_csv(perf_out, index=False)

    metric_order = ['Age', 'Gender', 'Ethnicity', 'Chief Complaint']
    dist_df['Metric'] = pd.Categorical(dist_df['Metric'], categories=metric_order, ordered=True)
    dist_df = dist_df.sort_values(by=['Month', 'Region', 'Metric'])
    dist_df.to_csv(dist_out, index=False)

    # Pivot for merging (regional level)
    dist_pivot_path = os.path.join(CLEAN_DIR, 'ecd_regional_distributions_pivoted.csv')
    pivot_df = dist_df.drop(columns=['Type', 'Metric']) \
        .melt(id_vars=['Month', 'Region']) \
        .pivot_table(index=['Month', 'Region'], columns='variable', values='value', aggfunc='mean') \
        .reset_index()
    pivot_df = pivot_df.sort_values(['Month', 'Region'])
    pivot_df.to_csv(dist_pivot_path, index=False)

    print(f"\n Saved regional performance  {perf_out}")
    print(f" Saved regional distributions  {dist_out}")
    print(f" Saved regional pivoted distributions  {dist_pivot_path}")

if __name__ == '__main__':
    main()
