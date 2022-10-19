# get the best validation img crop from a volume; it finds the crop with max number of foreground labels; assuming binary case and 3D
import numpy as np
import SimpleITK as sitk
import os
import cc3d

def load_img(filepath):
    sitkimg = sitk.ReadImage(filepath)
    arrimg = sitk.GetArrayFromImage(sitkimg)
    return arrimg


def get_best_val(arr):
    patchsize = [32,32,32]
    arrsize = arr.shape
    bestidx = [0,0,0]
    bestval = 0
    for z in range(0,arrsize[0],patchsize[0]):
        for y in range(0,arrsize[1],patchsize[1]):
            for x in range(0,arrsize[2],patchsize[2]):
                minivol = arr[z:z+patchsize[0],y:y+patchsize[1],x:x+patchsize[2]]
                cnts = np.unique(minivol, return_counts=True)[1]
                if cnts.shape[0] > 1:
                    minival = cnts[1]
                    if minival > bestval:
                        bestval = minival
                        bestidx = [z,y,x]
                        print(bestval)
    print(bestval)
    print(bestidx)


if __name__ == "__main__":
    srcpath = "/data/saumgupta/kidney-vessel/data/nnunet_ver2/nnUNet_raw_data_base/nnUNet_raw_data/Task270_KIDNEY/labelsTr/50H2_um.nii.gz"
    imgarr = load_img(srcpath)
    get_best_val(imgarr)