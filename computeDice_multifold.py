#compute DICE with all 3 GTs and fill into excel file

import SimpleITK as sitk
import numpy as np
import os, glob, sys
import torch
from torch.nn.functional import conv3d

foldpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/outputs/2d/topo"
gtroot = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/gt"
logfile = os.path.join(foldpath,"dice-20220210.csv")
sitkfilter = sitk.LabelOverlapMeasuresImageFilter()

def main():

    filelist = os.listdir(os.path.join(foldpath, "fold0"))
    filelist = [f for f in filelist if "nii" in f]
    filelist.sort()

    numFolds = [0,1,3]
    gtfolders = ['obs1','obs2_v1','obs2_v2']
    segclasses = [1,2]
    vallist = [0] * len(numFolds) * len(gtfolders) * len(segclasses)

    with open(logfile, 'a') as writefile:
        for i, filename in enumerate(filelist):
            writestr = filename
            for clind,clnum in enumerate(segclasses):
                for indg, gtf in enumerate(gtfolders):
                    gtpath = os.path.join(gtroot, os.path.join(gtf,filename))
                    for flind, foldnum in enumerate(numFolds):
                        filepath = os.path.join(foldpath, os.path.join("fold{}".format(foldnum), filename))

                        diceCoeff = calculateDiceCoeff(filepath, gtpath, clnum)
                        
                        vallist[clind*(len(numFolds)*len(gtfolders)) + len(numFolds)*indg + flind] += diceCoeff
                        
                        writestr = writestr + "," + str(diceCoeff)
            writestr = writestr + '\n'
            writefile.write(writestr)
            print(i)
        writestr = "Average"
        for loopvar in range(len(numFolds) * len(gtfolders) * len(segclasses)):
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