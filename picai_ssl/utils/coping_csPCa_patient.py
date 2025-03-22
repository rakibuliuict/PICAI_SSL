import os
import pandas as pd
import shutil
from glob import glob

# --- Step 1: Load CSV and extract patient IDs with case_csPCa == 'yes' ---
df = pd.read_csv(r"C:\Users\User\Downloads\picai_labels-main\picai_labels-main\clinical_information\marksheet.csv")

# Clean the 'case_csPCa' column
df['case_csPCa'] = df['case_csPCa'].astype(str).str.strip().str.lower()

# Filter rows
filtered_df = df[df['case_csPCa'] == 'yes']

# Check for correct patient ID column
if 'patient_id' in filtered_df.columns:
    patient_ids = filtered_df['patient_id'].astype(str).tolist()  # Convert to str for matching
else:
    print(" 'patient_id' column not found.")
    exit()

print(f" Found {len(patient_ids)} patients with case_csPCa == 'yes'")

# --- Step 2: Copy matching NIfTI files ---
source_folder = r"E:\Dataset\PICAI\workdir\workdir\nnUNet_raw_data\Task2201_picai_baseline\labelsTr"  # Replace with your actual images folder
destination_folder = r"E:\Dataset\PICAI\workdir\workdir\csPCa_patients\225_labels"  #  Replace with where you want to copy them

# Create output directory if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Get all NIfTI files in the folder
nifti_files = glob(os.path.join(source_folder, "*.nii.gz"))

copied_count = 0

for file_path in nifti_files:
    file_name = os.path.basename(file_path)
    prefix = file_name.split('_')[0]  # Extract patient ID from filename

    if prefix in patient_ids:
        shutil.copy(file_path, os.path.join(destination_folder, file_name))
        print(f"Copied: {file_name}")
        copied_count += 1

print(f"\n Done Total files copied: {copied_count}")
