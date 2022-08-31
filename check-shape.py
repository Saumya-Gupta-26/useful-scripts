import numpy as np
import SimpleITK as sitk
import os, glob

def find_vol(rootdir):
    vol_list = glob.glob(rootdir + '*/image/*.nii.gz')
    for filepath in vol_list:
        sitkimg = sitk.ReadImage(filepath)
        arrimg = sitk.GetArrayFromImage(sitkimg)
        print("{},{}".format(filepath, arrimg.shape))


def main():
    rootdir = "/data/saumgupta/slicer-tool/datasets/vessel-data/parse22/"
    find_vol(rootdir)



if __name__ == "__main__":
    main()
