#compute HD and ASSD with GT and fill into excel file
#for HD, report max of {hd(A,B), hd(B,A)}
#for ASSD, report pred->gt that is ASSD(pred, gt)

import SimpleITK as sitk
import numpy as np
import os, glob, sys
import torch
from torch.nn.functional import conv3d

from medpy.metric.binary import asd
from scipy.spatial.distance import directed_hausdorff

'''
foldpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/results/outputs/2d/crf"
gtroot = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/gt/obs2_v1"
'''
foldpath = "/data/saumgupta/miccai/dataset/nnunet-aorta/outputs/3d-ce-crf"
gtroot = "/home/saumya/aorta-segmentation/baseline/nnunet/data/data-format-for-all/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/labelsTr"
logfile = os.path.join(foldpath,"hd-assd-20220305.csv")
sitkfilter = sitk.LabelOverlapMeasuresImageFilter()

with open(logfile, 'w') as writefile:
    writefile.write("Filename,HD-lumen,ASSD-lumen,HD-media,ASSD-media\n")

def main():

    filelist = os.listdir(foldpath)
    filelist = [f for f in filelist if "nii" in f]
    filelist.sort()

    segclasses = [1,2]
    valHD = [0.0] * len(segclasses)
    valASSD = [0.0] * len(segclasses)

    with open(logfile, 'a') as writefile:
        for i, filename in enumerate(filelist):
            writestr = filename
            gtpath = os.path.join(gtroot, filename)
            filepath = os.path.join(foldpath, filename)

            arrayimage_gt = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(gtpath)))
            arrayimage_pred = np.squeeze(sitk.GetArrayFromImage(sitk.Cast(sitk.ReadImage(filepath), sitk.sitkUInt8)))

            #print(np.unique(arrayimage_pred, return_counts=True)) #0,1,2
            #print(np.unique(arrayimage_gt, return_counts=True)) #0,1,2

            for cLind, cL in enumerate(segclasses):
                temp_gt = np.where(arrayimage_gt == cL, 1.0, 0.0)
                temp_pred = np.where(arrayimage_pred == cL, 1.0, 0.0)

                #print(np.unique(temp_pred, return_counts=True)) #0.0,1.0
                #print(np.unique(temp_gt, return_counts=True)) #0.0,1.0

                hdCoeff = max(directed_hausdorff(temp_gt, temp_pred)[0], directed_hausdorff(temp_pred, temp_gt)[0])
                valHD[cLind] += hdCoeff

                assdCoeff = asd(temp_pred, temp_gt)
                valASSD[cLind] += assdCoeff

                writestr = writestr + "," + str(hdCoeff) + "," + str(assdCoeff)

            writestr = writestr + '\n'
            writefile.write(writestr)
            print(i)
        writestr = "Average," + str(valHD[0]/len(filelist)) + "," +  str(valASSD[0]/len(filelist)) + "," + str(valHD[1]/len(filelist)) + "," +  str(valASSD[1]/len(filelist))
        writefile.write(writestr)



if __name__ == "__main__":
    main()