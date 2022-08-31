import SimpleITK as sitk
import numpy as np
import os, glob, sys
import torch
from torch.nn.functional import conv3d

foldpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/outputs/2d/"
logfile = os.path.join(foldpath,"fold-violations-20220202.csv")

def main():

    filelist = os.listdir(os.path.join(foldpath, "fold0"))
    filelist = [f for f in filelist if "nii" in f]
    filelist.sort()

    numFolds = 4
    vallist = [0] * numFolds

    with open(logfile, 'a') as writefile:
        for i, filename in enumerate(filelist):
            maxfold = ""
            maxvalue = 0
            writestr = filename
            for foldnum in range(numFolds):
                filepath = os.path.join(foldpath, os.path.join("fold{}".format(foldnum), filename))
                percentViolations = calculateNumViolations(filepath)

                if percentViolations > maxvalue:
                    maxvalue = percentViolations
                    maxfold = "fold"+str(foldnum)
                
                vallist[foldnum] += percentViolations
                
                writestr = writestr + "," + str(percentViolations)
            writestr = writestr + "," + str(maxvalue) + "," + maxfold + '\n'
            writefile.write(writestr)
            print(i)
        writestr = "Average," + str(vallist[0]/len(filelist)) + "," + str(vallist[1]/len(filelist)) + "," + str(vallist[2]/len(filelist)) + "," + str(vallist[3]/len(filelist))  
        writefile.write(writestr)



def calculateNumViolations(filepath):
    interior = 1.0
    exterior = 2.0
    background = 3.0 # background + RV

    kernel = torch.ones((1,1,3,3,3))

    sitkimage_gt = sitk.ReadImage(filepath)
    imageArray = sitk.GetArrayFromImage(sitkimage_gt)
    
    torch_image = torch.from_numpy(np.expand_dims(np.expand_dims(imageArray,axis=0), axis=0)).double()

    torch_image = torch.where(torch_image == 0.0, background, torch_image)

    redZeros = torch.where(torch_image == interior, 0.0, interior)
    redZeros_greenOnes = torch.where(torch_image == exterior, 1.0, torch_image*redZeros)
    redZeros_greenZeros = torch.where(redZeros_greenOnes == 1.0, 0.0, torch_image*redZeros)
    del redZeros_greenOnes
    outputConv = conv3d(redZeros_greenZeros, kernel.double(), padding='same')
    #print(torch.unique(outputConv, return_counts=True))
    del redZeros_greenZeros
    maskOfRedCenters = torch.zeros_like(redZeros)
    del redZeros
    # create a mask get locations of redCenters.
    redOnes = torch.where(torch_image == interior, 1.0, 0.0)
    greenOnes = torch.where(torch_image == exterior, 1.0, 0.0)
    totalVoxels = redOnes.sum() + greenOnes.sum()
    del greenOnes
    maskOfRedCenters = redOnes
    # apply it on outputConv
    maskedOutputConv = outputConv * maskOfRedCenters
    del outputConv, maskOfRedCenters
    maskedOutputConv = torch.where(maskedOutputConv > 1.0, 1.0, 0.0)
    #print("locations of violating ones\n", maskedOutputConv)
    #print(torch.unique(maskedOutputConv, return_counts=True))
    
    # do dilation on those centers to get neighbors ----> approx superset of violations.
    supersetViolations = torch.where(conv3d(maskedOutputConv, kernel.float(), padding='same')>0,1.0,0.0)

    greenZeros = torch.where(torch_image == exterior, 0.0, 1.0)
    eqSix = torch.logical_and(supersetViolations, greenZeros).float()
    del greenZeros, supersetViolations
    eqSeven = torch.logical_and(redOnes, torch.logical_not(maskedOutputConv).float()).float()
    del redOnes, maskedOutputConv
    answer = torch.logical_and(eqSix, torch.logical_not(eqSeven).float()).float()
    del eqSix, eqSeven
    numViolations = answer.sum()

    percent_vio = numViolations.numpy()*100 / totalVoxels.numpy()
    return percent_vio



if __name__ == "__main__":
    main()