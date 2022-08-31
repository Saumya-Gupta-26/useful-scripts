#compute HD, ASSD with GTs and fill into excel file

import SimpleITK as sitk
import numpy as np
import os, glob, sys
import torch
from torch.nn.functional import conv3d

from medpy.metric.binary import asd
from scipy.spatial.distance import directed_hausdorff

foldpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/outputs/2d/topo"
gtroot = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/gt/obs2_v1"
logfile = os.path.join(foldpath,"hd-assd-20220223.csv")

with open(logfile, 'w') as writefile:
    writefile.write("Filename,HD-lumen-fold0,HD-lumen-fold1,HD-lumen-fold3,HD-media-fold0,HD-media-fold1,HD-media-fold2,ASSD-lumen-fold0,ASSD-lumen-fold1,ASSD-lumen-fold3,ASSD-media-fold0,ASSD-media-fold1,ASSD-media-fold2\n")

def main():

    filelist = os.listdir(os.path.join(foldpath, "fold0"))
    filelist = [f for f in filelist if "nii" in f]
    filelist.sort()

    numFolds = [0,1,3]
    segclasses = [1,2]

    with open(logfile, 'a') as writefile:
        for i, filename in enumerate(filelist):
            writestr = filename
            gtpath = os.path.join(gtroot, filename)
            arrayimage_gt = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(gtpath)))

            for metric in range(2): # 0 = HD; 1 = ASSD
                for clnum in segclasses:
                    for foldnum in numFolds:
                        filepath = os.path.join(foldpath, os.path.join("fold{}".format(foldnum), filename))

                        arrayimage_pred = np.squeeze(sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(filepath), sitk.sitkUInt8)))
                        
                        temp_gt = np.where(arrayimage_gt == clnum, 1.0, 0.0)
                        temp_pred = np.where(arrayimage_pred == clnum, 1.0, 0.0)

                        if metric == 0:
                            Coeff = max(directed_hausdorff(temp_gt, temp_pred)[0], directed_hausdorff(temp_pred, temp_gt)[0])
                        elif metric == 1:
                            Coeff = asd(temp_pred, temp_gt)

                        writestr = writestr + "," + str(Coeff)

            writestr = writestr + '\n'
            writefile.write(writestr)
            print(i)
        writestr = "Average"
        writefile.write(writestr)


if __name__ == "__main__":
    main()