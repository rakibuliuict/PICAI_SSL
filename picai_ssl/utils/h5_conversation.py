import os
import h5py
import nibabel as nib
import numpy as np

# Base path where patient folders are organized


organized_base = r"/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset"  #  Update with your actual path
h5_output_folder = os.path.join(organized_base, "h5")
os.makedirs(h5_output_folder, exist_ok=True)

# List all patient IDs (folder names)
patient_ids = [name for name in os.listdir(organized_base)
               if os.path.isdir(os.path.join(organized_base, name)) and name.isdigit()]

for pid in patient_ids:
    try:
        patient_folder = os.path.join(organized_base, pid)
        t2w_path = os.path.join(patient_folder, "t2w", f"{pid}_t2w.nii.gz")
        adc_path = os.path.join(patient_folder, "adc", f"{pid}_adc.nii.gz")
        hbv_path = os.path.join(patient_folder, "hbv", f"{pid}_hbv.nii.gz")
        seg_path = os.path.join(patient_folder, "seg", f"{pid}_seg.nii.gz")

        # Load images using nibabel
        t2w = nib.load(t2w_path).get_fdata()
        adc = nib.load(adc_path).get_fdata()
        hbv = nib.load(hbv_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()

        # Normalize image modalities
        def norm(img): return (img - np.mean(img)) / np.std(img)
        t2w, adc, hbv = map(norm, [t2w, adc, hbv])

        # Save to HDF5
        h5_path = os.path.join(h5_output_folder, f"{pid}.h5")
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset("image/t2w", data=t2w, compression="gzip")
            hf.create_dataset("image/adc", data=adc, compression="gzip")
            hf.create_dataset("image/hbv", data=hbv, compression="gzip")
            hf.create_dataset("label/seg", data=seg.astype(np.uint8), compression="gzip")

        print(f" Converted {pid} → {h5_path}")

    except Exception as e:
        print(f" Error with patient {pid}: {e}")

print("\n All patient folders converted to .h5 format!")
