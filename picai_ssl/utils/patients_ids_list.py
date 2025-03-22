import pandas as pd

# Load the CSV file
df = pd.read_csv(r"C:\Users\User\Downloads\picai_labels-main\picai_labels-main\clinical_information\marksheet.csv")

# Clean 'case_csPCa' column
df['case_csPCa'] = df['case_csPCa'].astype(str).str.strip().str.lower()

# Filter rows where case_csPCa == 'yes'
filtered_df = df[df['case_csPCa'] == 'yes']

# Extract patient IDs
if 'patient_id' in df.columns:
    patient_ids = filtered_df['patient_id'].tolist()
else:
    print(" Column 'patient_id' not found. Available columns:", df.columns.tolist())
    patient_ids = []

# Print result
print("\n Patients with case_csPCa == 'Yes':")
print(patient_ids)

# # Save to CSV
# output_df = pd.DataFrame({'patient_id': patient_ids})
# output_path = r"C:\Users\User\Downloads\picai_labels-main\picai_labels-main\csPCa_patient_ids.csv"
# output_df.to_csv(output_path, index=False)

# print(f"\n Saved patient_ids to: {output_path}")
