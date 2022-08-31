import os, sys, glob

dst_root = "/data/saumgupta/slicer-tool/datasets/vessel-data/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task101_VESSEL/labelsTr"

allfiles = os.listdir(dst_root)
jsonstr = ""
for af in allfiles:
    dstpath = os.path.join(dst_root, af)
    jsonstr = jsonstr + ",{\"image\":\"" + dstpath.replace('labelsTr', 'imagesTr') + "\",\"label\":\"" + dstpath + "\"}"
print(jsonstr)
