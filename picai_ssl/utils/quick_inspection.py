import h5py

path = '/content/drive/MyDrive/SemiSL/Dataset/PICAI_dataset/10005/10005.h5'
with h5py.File(path, 'r') as f:
    print("Keys at root level:", list(f.keys()))
    for key in f.keys():
        print(f"{key} --> {type(f[key])}")
