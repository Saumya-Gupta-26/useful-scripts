#compute DICE with GT and fill into excel file

import SimpleITK as sitk
import numpy as np
import os, glob, sys
import torch
from torch.nn.functional import conv3d
'''
foldpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/results/outputs/2d/crf"
gtroot = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/gt/obs2_v1"
'''
foldpath = "/data/saumgupta/miccai/dataset/nnunet-aorta/outputs/3d-ce-crf"
gtroot = "/home/saumya/aorta-segmentation/baseline/nnunet/data/data-format-for-all/nnUNet_raw_data_base/nnUNet_raw_data/Task500_Aorta/labelsTr"

logfile = os.path.join(foldpath,"dice-20220305.csv")
sitkfilter = sitk.LabelOverlapMeasuresImageFilter()

def main():

    filelist = os.listdir(foldpath)
    filelist = [f for f in filelist if "nii" in f]
    filelist.sort()

    segclasses = [1,2]
    vallist = [0.0] * len(segclasses)

    with open(logfile, 'a') as writefile:
        for i, filename in enumerate(filelist):
            writestr = filename
            gtpath = os.path.join(gtroot, filename)
            filepath = os.path.join(foldpath, filename)

            for cLind, cL in enumerate(segclasses):
                diceCoeff = calculateDiceCoeff(filepath, gtpath, cL)
                vallist[cLind] += diceCoeff
                writestr = writestr + "," + str(diceCoeff)

            writestr = writestr + '\n'
            writefile.write(writestr)
            print(i)
        writestr = "Average"
        for loopvar in range(len(segclasses)):
            writestr = writestr + "," + str(vallist[loopvar]/len(filelist))
        writefile.write(writestr)


def calculateDiceCoeff(filepath_pred, filepath_gt, clnum):
    sitkimage_gt = sitk.ReadImage(filepath_gt)
    sitkimage_pred = sitk.Cast(sitk.ReadImage(filepath_pred), sitk.sitkUInt8)

    sitkfilter.Execute(sitkimage_gt, sitkimage_pred)
    val = sitkfilter.GetDiceCoefficient(clnum) #class-number
    return val


if __name__ == "__main__":
    main()