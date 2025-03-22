import os
import shutil
from glob import glob

# Paths (update these!)
source_images = r"E:\Dataset\PICAI\workdir\workdir\csPCa_patients\225_images"         #  Folder with scans
source_labels = r"E:\Dataset\PICAI\workdir\workdir\csPCa_patients\225_labels"         #  Folder with label masks
output_base = r"E:\Dataset\PICAI\workdir\workdir\csPCa_patients\PICAI_dataset" #  Destination folder

# Modality mapping based on suffix
suffix_to_modality = {
    '0000': 't2w',
    '0001': 'adc',
    '0002': 'hbv'
}

# Create output base directory
os.makedirs(output_base, exist_ok=True)

# --- Step 1: Organize Scans ---
nifti_files = glob(os.path.join(source_images, "*.nii.gz"))

for file_path in nifti_files:
    file_name = os.path.basename(file_path)

    try:
        parts = file_name.split('_')
        patient_id = parts[0]
        suffix = parts[-1].replace(".nii.gz", "")
        modality = suffix_to_modality.get(suffix)

        if modality is None:
            print(f" Skipping unknown modality: {file_name}")
            continue

        # Create patient/modality directory
        dest_folder = os.path.join(output_base, patient_id, modality)
        os.makedirs(dest_folder, exist_ok=True)

        # Rename & copy scan
        new_file_name = f"{patient_id}_{modality}.nii.gz"
        dest_path = os.path.join(dest_folder, new_file_name)
        shutil.copy(file_path, dest_path)
        print(f" Copied scan: {file_name} → {dest_path}")

    except Exception as e:
        print(f" Error processing scan {file_name}: {e}")

# --- Step 2: Organize Labels ---
label_files = glob(os.path.join(source_labels, "*.nii.gz"))

for label_path in label_files:
    label_name = os.path.basename(label_path)
    patient_id = os.path.splitext(label_name)[0].split('_')[0]  # Handles 10005.nii.gz or 10005_label.nii.gz

    try:
        # Create seg folder inside patient directory
        seg_folder = os.path.join(output_base, patient_id, "seg")
        os.makedirs(seg_folder, exist_ok=True)

        # Standard name: 10005_seg.nii.gz
        dest_label_path = os.path.join(seg_folder, f"{patient_id}_seg.nii.gz")
        shutil.copy(label_path, dest_label_path)
        print(f" Copied label: {label_name} → {dest_label_path}")

    except Exception as e:
        print(f" Error processing label {label_name}: {e}")

print("\n Done organizing images and labels!")
