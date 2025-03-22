import os
import shutil
from glob import glob
import nibabel as nib
import numpy as np
import h5py

base_dir = "/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"

modalities = {
    't2w': '_t2w.nii.gz',
    'adc': '_adc.nii.gz',
    'hbv': '_hbv.nii.gz',
    'seg': '_seg.nii.gz'
}

for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    print(f"\n📁 Processing patient: {patient_id}")

    files_ready = True

    # Step 1: Rename modality files if needed
    for modality, suffix in modalities.items():
        matches = glob(os.path.join(patient_path, f"*{suffix}"))
        if not matches:
            print(f"⚠️ Missing file for {modality}")
            files_ready = False
            break  # Stop checking further if any is missing

        src = matches[0]
        dst = os.path.join(patient_path, f"{modality}.nii.gz")
        if not os.path.exists(dst):
            shutil.move(src, dst)
            print(f"✅ Renamed: {os.path.basename(src)} → {modality}.nii.gz")
        else:
            print(f"🟡 Already renamed: {modality}.nii.gz")

    # Step 2: Proceed to H5 creation only if all files are ready
    if not files_ready:
        print(f"⛔ Skipping .h5 creation for {patient_id} due to missing files.")
        continue

    try:
        t2w = nib.load(os.path.join(patient_path, "t2w.nii.gz")).get_fdata()
        adc = nib.load(os.path.join(patient_path, "adc.nii.gz")).get_fdata()
        hbv = nib.load(os.path.join(patient_path, "hbv.nii.gz")).get_fdata()
        seg = nib.load(os.path.join(patient_path, "seg.nii.gz")).get_fdata()

        def norm(img): return (img - np.mean(img)) / np.std(img)
        t2w, adc, hbv = map(norm, [t2w, adc, hbv])

        h5_path = os.path.join(patient_path, f"{patient_id}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("image/t2w", data=t2w, compression="gzip")
            hf.create_dataset("image/adc", data=adc, compression="gzip")
            hf.create_dataset("image/hbv", data=hbv, compression="gzip")
            hf.create_dataset("label/seg", data=seg.astype(np.uint8), compression="gzip")

        print(f"✅ Created H5 file: {h5_path}")

    except Exception as e:
        print(f"❌ Error processing {patient_id}: {e}")

print("\n✅ All complete! Only valid patients were processed to H5.")
