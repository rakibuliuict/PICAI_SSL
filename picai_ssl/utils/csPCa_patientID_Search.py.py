import pandas as pd

# Load the uploaded CSV file
df = pd.read_csv(r"C:\Users\User\Downloads\picai_labels-main\picai_labels-main\clinical_information\marksheet.csv")

# Check the unique values in 'case_csPCa' column (to debug formatting issues)
print("Unique values in 'case_csPCa':", df['case_csPCa'].unique())

# Clean the 'case_csPCa' column: remove extra spaces, convert to lowercase
df['case_csPCa'] = df['case_csPCa'].astype(str).str.strip().str.lower()

# Filter rows where case_csPCa == 'yes'
filtered_df = df[df['case_csPCa'] == 'yes']

# Extract patient IDs
if 'patient_id' in df.columns:
    patient_ids = filtered_df['patient_id'].tolist()
else:
    print("Column 'patient ID' not found. Available columns are:", df.columns.tolist())
    patient_ids = []

# Print result
print("\n Patients with case_csPCa == 'Yes':")
print(len(patient_ids))
