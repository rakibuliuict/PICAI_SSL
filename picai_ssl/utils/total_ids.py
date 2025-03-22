import os

# Path to the folder containing patient folders
source_folder = "/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"  # 👈 Update this
# Path where you want to save the patient_ids.txt file
output_folder = "/content/drive/MyDrive/SemiSL/Code/PICAI_SSL/Basecode/Datasets/picai/data_split"  # 👈 Update this

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Collect all subdirectories as patient IDs
patient_ids = [name for name in os.listdir(source_folder)
               if os.path.isdir(os.path.join(source_folder, name))]

# Sort for consistency
patient_ids.sort()

# Define output file path
output_file = os.path.join(output_folder, "patient_ids.txt")

# Save to text file
with open(output_file, "w") as f:
    for pid in patient_ids:
        f.write(f"{pid}\n")

print(f"✅ Saved {len(patient_ids)} patient IDs to:\n{output_file}")
