# goal: visualize the difference between GT and prediction
## packages : https://pypi.org/project/connected-components-3d/
### compute difference
### run connected components (CC)
### each CC is a different colour

# variations
## Only keep diff > 0 that is, structures in GT missing in prediction (false negatives)
## Keep only structures having atleast N number of pixels
## And add the GT as well. Thus, structures in GT & in pred would be one color (val=1)

import numpy as np
import SimpleITK as sitk
import os
import cc3d

def load_img(filepath):
    sitkimg = sitk.ReadImage(filepath)
    arrimg = sitk.GetArrayFromImage(sitkimg)
    return arrimg

def print_stats(alist):
    arr = np.array(alist)
    print("Min area: {}".format(np.min(arr)))
    print("Max area: {}".format(np.max(arr)))
    print("Mean area: {}".format(np.mean(arr)))
    print("Median area: {}".format(np.median(arr)))
    print(alist)

def conn_comp(arr):
    labels_out, numcomp = cc3d.connected_components(arr, connectivity=26, return_N=True) # 26-connected
    print("Number of components: {}".format(numcomp))

    area_list = []

    instance_seg = np.zeros_like(labels_out)

    for label, image in cc3d.each(labels_out, binary=False, in_place=True):
        area = np.sum(image)/label
        area_list.append(area)
        if (area > 5.0):
            instance_seg += image

    #print_stats(area_list)

    return instance_seg


def save_img(arr, sitkrefpath):
    sitkimg = sitk.GetImageFromArray(arr)

    sitkrefimg = sitk.ReadImage(sitkrefpath)
    sitkimg.SetSpacing(sitkrefimg.GetSpacing())
    sitkimg.SetOrigin(sitkrefimg.GetOrigin())
    sitkimg.SetDirection(sitkrefimg.GetDirection())
    sitk.WriteImage(sitkimg, sitkrefpath.replace("label","diffGT_A5").replace("predict", "diffGT_A5"))

def main():
    rootdir = "/data/saumgupta/slicer-tool/datasets/vessel-data/outputs"
    src1 = os.path.join(rootdir, "swinunetrl-parse22-dice-label_0.nii.gz")
    src2 = os.path.join(rootdir, "swinunetrl-parse22-dice-predict_0.nii.gz")
    arr1 = load_img(src1)
    arr2 = load_img(src2)

    diff = np.absolute(np.subtract(arr1,arr2))
    #diff = np.subtract(arr1,arr2) # False-negatives
    #diff = diff > 0.0
    diff = diff.astype(np.uint8)
    instance_seg = conn_comp(diff)

    
    mask = instance_seg > 0.0
    mask = mask.astype(np.uint8)
    instance_seg += mask # Increase all CC's value by 1
    instance_seg += arr1 # Add GT to visualize whole

    save_img(instance_seg, src1)



if __name__ == "__main__":
    main()
