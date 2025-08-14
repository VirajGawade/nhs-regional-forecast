import pandas as pd
import csv

# config
INPUT_CSV = r"C:\Users\viraj\Downloads\nhs regional scripts\nomis_regional_raw.csv"
OUTPUT_CSV = r"C:\Users\viraj\Downloads\nhs regional scripts\nomis_regional_cleaned.csv"


def parse_nomis_file(filepath):
    """Extracts gendered regional blocks from raw Nomis regional CSV."""
    blocks = []
    gender = None
    age = None
    table_header = None
    table_rows = []
    state = "searching"

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [cell.strip() for cell in row]

            # Skip empty or notes
            if not any(row) or "Figures may not sum" in row[0]:
                continue

            if row[0].startswith("Gender"):
                gender = row[1]
            elif row[0].startswith("Age"):
                age = row[1]
            elif row[0] == "Area":
                table_header = row
                table_rows = []
                state = "reading"
            elif state == "reading":
                if row[0].startswith("gor:"):
                    table_rows.append(row)
                elif row[0].startswith("country:"):  # skip national row
                    continue
                else:
                    if table_rows:
                        blocks.append((gender, age, table_header, table_rows))
                        table_rows = []
                        state = "searching"

        # Final block if file ends without blank lines
        if table_rows:
            blocks.append((gender, age, table_header, table_rows))

    return blocks

def clean_blocks(blocks):
    """Converts parsed blocks to long-form cleaned DataFrame."""
    all_frames = []
    for gender, age, header, rows in blocks:
        if gender != "Total":
            continue  # Only keep Gender = Total

        df = pd.DataFrame(rows, columns=header)
        df["Region"] = df["Area"].str.replace("gor:", "", regex=False)
        df = df.drop(columns=["Area"])

        df_melted = df.melt(id_vars="Region", var_name="Year", value_name="Population")
        all_frames.append(df_melted)

    if not all_frames:
        return pd.DataFrame(columns=["Region", "Year", "Population"])  # Empty fallback

    df_final = pd.concat(all_frames, ignore_index=True)
    df_final["Year"] = pd.to_numeric(df_final["Year"], errors="coerce").astype("Int64")
    df_final["Population"] = pd.to_numeric(df_final["Population"], errors="coerce")
    df_final = df_final.dropna(subset=["Population", "Year"])

    return df_final[["Region", "Year", "Population"]].sort_values(by=["Region", "Year"])


if __name__ == "__main__":
    blocks = parse_nomis_file(INPUT_CSV)
    if not blocks:
        raise ValueError(" No usable blocks found in the input file.")

    df_clean = clean_blocks(blocks)
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f" Cleaned regional Nomis data (Gender=Total) saved to: {OUTPUT_CSV}")
