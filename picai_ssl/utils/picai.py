
import os
import shutil
from glob import glob
import nibabel as nib
import numpy as np
import h5py

# Base path where patient folders are stored
base_dir = "/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"

# Modality mapping
modalities = {
    't2w': '_t2w.nii.gz',
    'adc': '_adc.nii.gz',
    'hbv': '_hbv.nii.gz',
    'seg': '_seg.nii.gz'
}

# Loop through all patient folders
for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    print(f"\n📁 Processing patient: {patient_id}")

    data = {}

    # Step 1: Rename files if needed
    for modality, suffix in modalities.items():
        matches = glob(os.path.join(patient_path, f"*{suffix}"))
        if not matches:
            print(f"⚠️ Missing file for {modality}")
            break

        src = matches[0]
        dst = os.path.join(patient_path, f"{modality}.nii.gz")

        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"✅ Renamed: {os.path.basename(src)} → {modality}.nii.gz")
        else:
            print(f"🟡 Already renamed: {modality}.nii.gz")

    # Step 2: Load all files into memory
    try:
        t2w = nib.load(os.path.join(patient_path, "t2w.nii.gz")).get_fdata()
        adc = nib.load(os.path.join(patient_path, "adc.nii.gz")).get_fdata()
        hbv = nib.load(os.path.join(patient_path, "hbv.nii.gz")).get_fdata()
        seg = nib.load(os.path.join(patient_path, "seg.nii.gz")).get_fdata()

        # Normalize image data
        def norm(img): return (img - np.mean(img)) / np.std(img)
        t2w, adc, hbv = map(norm, [t2w, adc, hbv])

        # Save to H5
        h5_path = os.path.join(patient_path, f"{patient_id}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("image/t2w", data=t2w, compression="gzip")
            hf.create_dataset("image/adc", data=adc, compression="gzip")
            hf.create_dataset("image/hbv", data=hbv, compression="gzip")
            hf.create_dataset("label/seg", data=seg.astype(np.uint8), compression="gzip")

        print(f"✅ Created H5 file: {h5_path}")

    except Exception as e:
        print(f"❌ Error processing {patient_id}: {e}")

print("\n🎉 All patients processed with .h5 files saved in their folders.")
