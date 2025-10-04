# src/01_clean_dataset.py
import os
import pandas as pd

# Input folder (yaha tera extracted folder ka path daal)
base_path = r"C:\Users\Krishna\Downloads\OSD-101_metadata_OSD-101-ISA"

# Important files
investigation_file = os.path.join(base_path, "i_Investigation.txt")
study_file = os.path.join(base_path, "s_OSD-101.txt")  # study file ka naam check kar lena
assay_files = [f for f in os.listdir(base_path) if f.startswith("a_")]

# Read investigation file
inv_df = pd.read_csv(investigation_file, sep="\t", dtype=str, on_bad_lines="skip")
print("✅ Investigation loaded:", investigation_file)

# Read study file
study_df = pd.read_csv(study_file, sep="\t", dtype=str, on_bad_lines="skip")
print("✅ Study loaded:", study_file)

# Merge assays
assay_dfs = []
for f in assay_files:
    path = os.path.join(base_path, f)
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, on_bad_lines="skip")
        assay_dfs.append(df)
        print("✅ Assay loaded:", f)
    except Exception as e:
        print("⚠ Could not load assay:", f, "| Error:", e)

# Combine everything
all_dfs = [inv_df, study_df] + assay_dfs
clean_df = pd.concat(all_dfs, axis=0, ignore_index=True)

# Fill empty columns
if "Summary" not in clean_df.columns:
    clean_df["Summary"] = clean_df.apply(lambda x: " ".join(x.dropna().astype(str)), axis=1)

# Save cleaned dataset
out_file = os.path.join(base_path, "OSD-101_clean.csv")
clean_df.to_csv(out_file, index=False)

print(f"🎉 Clean dataset saved at {out_file}")
