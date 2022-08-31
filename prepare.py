import os, glob
import shutil

srclist = glob.glob("/data/saumgupta/slicer-tool/datasets/vessel-data/parse22/*/label/*.nii.gz")
dstdir = "/data/saumgupta/slicer-tool/datasets/vessel-data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task101_VESSEL/labelsTr"

for fp in srclist:
    newname = fp.split('/')[-1].replace("PA", "PA_")#.replace(".nii.gz","_0000.nii.gz")
    shutil.copy(fp, os.path.join(dstdir, newname))