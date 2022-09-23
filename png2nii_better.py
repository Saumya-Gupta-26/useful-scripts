import glob 
import SimpleITK as sitk
import numpy as np
from PIL import Image, ImageOps
def convert(src_dir, binary=False):
    filelist = glob.glob(src_dir+'/*.png')
    print("Todo : {}".format(len(filelist)))

    for ind,fpath in enumerate(filelist):
        ### read png; converting input image to grayscale (0-255)
        if binary:
            arrayimage = np.expand_dims(np.squeeze(np.asarray(ImageOps.grayscale(Image.open(fpath)))),0)
        else:
            arrayimage = np.expand_dims(np.asarray(Image.open(fpath)),0)
        print(arrayimage.shape)

        ### covert to sitk-format
        sitkimage = sitk.GetImageFromArray(arrayimage)

        ### Save volume to file
        output_filename = fpath.replace(".png", ".nii.gz")
        sitk.WriteImage(sitkimage, output_filename)

        print(ind)

if __name__ == "__main__":
    src_dir = "/scr/saumgupta/user-study-int-seg/saumya_user_study/pixel"
    convert(src_dir, binary=True)