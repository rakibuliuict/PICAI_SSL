
import os
import shutil
from glob import glob

# Base directory containing all patient folders (e.g., G:\)
base_dir = r"/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"  #  Change this to your base folder

# Expected modalities
modalities = ['t2w', 'adc', 'hbv', 'seg']

for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    print(f"\n Processing {patient_id}")

    for modality in modalities:
        modality_folder = os.path.join(patient_path, modality)

        if not os.path.exists(modality_folder):
            print(f" Skipping missing folder: {modality_folder}")
            continue

        # Find any .nii.gz file inside the modality folder
        nii_files = glob(os.path.join(modality_folder, "*.nii.gz"))
        if not nii_files:
            print(f" No .nii.gz file found in {modality_folder}")
            continue

        src_file = nii_files[0]
        dest_file = os.path.join(patient_path, f"{modality}.nii.gz")

        # Move and rename the file
        shutil.move(src_file, dest_file)
        print(f" Moved {os.path.basename(src_file)} → {modality}.nii.gz")

        # Remove empty modality folder
        try:
            os.rmdir(modality_folder)
        except Exception as e:
            print(f" Could not remove folder {modality_folder}: {e}")

print("\n🎉 All patient folders successfully flattened.")
