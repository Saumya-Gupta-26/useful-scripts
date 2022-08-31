# convert bmp to nii.gz
import SimpleITK as sitk
import os, sys, glob
import numpy as np
from PIL import Image

#src_dir = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/Data_set_B/LABELS_MASK"
#dst_dir = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task800_IVUS/labelsTr"

src_dir = "/data/saumgupta/miccai/dataset/IVUS/Test_Set/Data_set_B/png/LABELS_obs2_v2"
dst_dir = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/gt/obs2_v2"

#lum_frame_03_0010_003.txt
def image_only():
    gtfiles = glob.glob(src_dir+'/lum*.txt')
    print("Todo : {}".format(len(gtfiles)))

    for ind,fpath_gt in enumerate(gtfiles):
        ### read png; converting input image to grayscale (0-255)
        arrayimage = np.expand_dims(np.asarray(Image.open(fpath_gt.replace('LABELS_obs1','DCM').replace('Test_Set','Training_Set').replace('lum_','').replace('.txt','.png'))),0)

        ### covert to sitk-format
        sitkimage = sitk.GetImageFromArray(arrayimage)

        ### Save volume to file
        fname_gt = fpath_gt.split('/')[-1].replace('.txt','.nii.gz').replace('lum_','')
        nsplits = fname_gt.split('_')
        fname_gt = nsplits[0] + nsplits[1] + '_' + nsplits[2] + nsplits[3]
        
        output_filename = os.path.join(dst_dir,fname_gt.replace('.nii.gz', '_0000.nii.gz'))
        sitk.WriteImage(sitkimage, output_filename)
        print(ind)


def gt_only():
    gtfiles = glob.glob(src_dir+'/*_gt.png')
    print("Todo : {}".format(len(gtfiles)))

    for ind,fpath_gt in enumerate(gtfiles):
        ### read png; converting input image to grayscale (0-255)
        arrayimage_gt = np.expand_dims(np.asarray(Image.open(fpath_gt)),0)

        ### covert to sitk-format
        sitkimage_gt = sitk.GetImageFromArray(arrayimage_gt)

        ### Save volume to file
        fname_gt = fpath_gt.split('/')[-1].replace('_gt.png','.nii.gz')
        nsplits = fname_gt.split('_')
        fname_gt = nsplits[0] + nsplits[1] + '_' + nsplits[2] + nsplits[3]
        output_filename_gt = os.path.join(dst_dir,fname_gt)
        sitk.WriteImage(sitkimage_gt, output_filename_gt)

        print(ind)



# INPUT and SEG VOLUMES NEED TO BE 3D -- so we do np.expand_dims

#frame_01_0001_003_gt.png ---> X_Y.nii.gz
#frame_01_0001_003.png ---> X_Y_0000.nii.gz
def image_and_gt():
    gtfiles = glob.glob(src_dir+'/*.png')
    print("Todo : {}".format(len(gtfiles)))

    for ind,fpath_gt in enumerate(gtfiles):
        ### read png; converting input image to grayscale (0-255)
        arrayimage_gt = np.expand_dims(np.asarray(Image.open(fpath_gt)),0)
        arrayimage = np.expand_dims(np.asarray(Image.open(fpath_gt.replace('LABELS_MASK','DCM').replace('_gt',''))),0)

        ### covert to sitk-format
        sitkimage_gt = sitk.GetImageFromArray(arrayimage_gt)
        sitkimage = sitk.GetImageFromArray(arrayimage)

        sitkimage.SetSpacing(sitkimage_gt.GetSpacing())
        sitkimage.SetOrigin(sitkimage_gt.GetOrigin())
        sitkimage.SetDirection(sitkimage_gt.GetDirection())

        ### Save volume to file
        fname_gt = fpath_gt.split('/')[-1].replace('_gt.png','.nii.gz')
        nsplits = fname_gt.split('_')
        fname_gt = nsplits[0] + nsplits[1] + '_' + nsplits[2] + nsplits[3]
        output_filename_gt = os.path.join(dst_dir,fname_gt)
        sitk.WriteImage(sitkimage_gt, output_filename_gt)
        
        output_filename = output_filename_gt.replace('labelsTr','imagesTr').replace('.nii.gz', '_0000.nii.gz')
        sitk.WriteImage(sitkimage, output_filename)

        print(ind)

if __name__ == "__main__":
    #image_and_gt()
    #image_only()
    gt_only()