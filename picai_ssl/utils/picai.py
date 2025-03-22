import os
import nibabel as nib
import numpy as np
import h5py

# Path to the dataset root
base_dir = "/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"  # 🔁 Update if needed

# Loop through each patient folder
for patient_id in os.listdir(base_dir):
    patient_path = os.path.join(base_dir, patient_id)
    if not os.path.isdir(patient_path):
        continue

    print(f"\n📁 Processing {patient_id}")

    # Define paths to each modality
    t2w_path = os.path.join(patient_path, "t2w.nii.gz")
    adc_path = os.path.join(patient_path, "adc.nii.gz")
    hbv_path = os.path.join(patient_path, "hbv.nii.gz")
    seg_path = os.path.join(patient_path, "seg.nii.gz")

    # Check if all required files exist
    if not all(os.path.exists(p) for p in [t2w_path, adc_path, hbv_path, seg_path]):
        print(f"⚠️ Missing one or more files for {patient_id}, skipping.")
        continue

    try:
        # Load images
        t2w = nib.load(t2w_path).get_fdata()
        adc = nib.load(adc_path).get_fdata()
        hbv = nib.load(hbv_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # Normalize images
        def norm(img): return (img - np.mean(img)) / np.std(img)
        t2w, adc, hbv = map(norm, [t2w, adc, hbv])

        # Create h5 file
        h5_path = os.path.join(patient_path, f"{patient_id}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("image/t2w", data=t2w, compression="gzip")
            hf.create_dataset("image/adc", data=adc, compression="gzip")
            hf.create_dataset("image/hbv", data=hbv, compression="gzip")
            hf.create_dataset("label/seg", data=seg.astype(np.uint8), compression="gzip")

        print(f"✅ Saved H5 file: {h5_path}")

    except Exception as e:
        print(f"❌ Failed to process {patient_id}: {e}")

print("\n🎉 All valid patients processed into .h5 files.")
