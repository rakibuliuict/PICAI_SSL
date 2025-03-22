import random
import math

# Path to your input file (update as needed)
input_file = "/content/drive/MyDrive/SemiSL/Code/PICAI_SSL/Basecode/Datasets/picai/data_split/patient_ids.txt"

# Output paths
train_file = "/content/drive/MyDrive/SemiSL/Code/PICAI_SSL/Basecode/Datasets/picai/data_split/train.txt"
test_file = "/content/drive/MyDrive/SemiSL/Code/PICAI_SSL/Basecode/Datasets/picai/data_split/test.txt"

# Step 1: Load all patient IDs from the file
with open(input_file, "r") as f:
    patient_ids = [line.strip() for line in f if line.strip()]

# Step 2: Shuffle and split (80/20)
random.shuffle(patient_ids)
split_index = math.floor(0.8 * len(patient_ids))
train_ids = patient_ids[:split_index]
test_ids = patient_ids[split_index:]

# Step 3: Save to files
with open(train_file, "w") as f:
    for pid in train_ids:
        f.write(f"{pid}\n")

with open(test_file, "w") as f:
    for pid in test_ids:
        f.write(f"{pid}\n")

print(f"✅ Done! {len(train_ids)} IDs in train.txt and {len(test_ids)} IDs in test.txt")
print(f"📁 Saved:\n  → {train_file}\n  → {test_file}")
