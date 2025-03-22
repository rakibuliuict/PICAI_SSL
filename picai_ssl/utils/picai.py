import os
import shutil
from glob import glob

# Base folder containing patient subfolders
base_path = r"/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"  #  Change this to your actual base directory

# Loop through each patient folder
for patient_id in os.listdir(base_path):
    patient_folder = os.path.join(base_path, patient_id)
    if not os.path.isdir(patient_folder):
        continue

    try:
        print(f"\n Processing patient: {patient_id}")

        # Define expected subfolders
        modalities = ['t2w', 'adc', 'hbv', 'seg']
        for modality in modalities:
            modality_folder = os.path.join(patient_folder, modality)
            if not os.path.exists(modality_folder):
                print(f" Missing modality folder: {modality}")
                continue

            # Find .nii.gz file in the modality folder
            nii_files = glob(os.path.join(modality_folder, "*.nii.gz"))
            if len(nii_files) == 0:
                print(f" No .nii.gz file in {modality_folder}")
                continue

            # Move and rename the file
            src_file = nii_files[0]
            dest_file = os.path.join(patient_folder, f"{modality}.nii.gz")
            shutil.move(src_file, dest_file)
            print(f" Moved {modality}: {os.path.basename(src_file)} → {os.path.basename(dest_file)}")

            # Remove empty subfolder
            os.rmdir(modality_folder)

        print(f" Done flattening patient folder: {patient_id}")

    except Exception as e:
        print(f" Error processing {patient_id}: {e}")

print("\n All patient folders flattened successfully!")
