'''
use --save_npz in predict command of nnunet and it will save softmax probabilities in .npz format
'''
from PIL import Image
import numpy as np
import cv2
inputpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/results/screenshot-likelihoodmap/frame07_0005003.npz"

dstpath = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/nnunet/test-data/results/screenshot-likelihoodmap"

myarray_obj = np.load(inputpath, allow_pickle=True)
'''
for key, myarray in myarray_obj.items():
    #print(np.min(myarray), np.max(myarray)) #0.0, 1.0

    myarray = np.squeeze(myarray) # 3,384,384
    slice1 = (myarray[0] * 255).astype(np.uint8)
    slice2 = (myarray[1] * 255).astype(np.uint8)
    slice3 = (myarray[2] * 255).astype(np.uint8)
    slice4 =  (np.amax(myarray, axis=0) * 255).astype(np.uint8)

    img_bg = Image.fromarray(slice1)
    img_lumen = Image.fromarray(slice2)
    img_media = Image.fromarray(slice3)
    img_combined = Image.fromarray(slice4)

    img_bg.save('img_bg.png')
    img_lumen.save('img_lumen.png')
    img_media.save('img_media.png')
    img_combined.save('img_combined.png')



for key, myarray in myarray_obj.items():
    #print(np.min(myarray), np.max(myarray)) #0.0, 1.0

    myarray = np.squeeze(myarray) # 3,384,384
    slice1 = (myarray[0] * 65535).astype(np.uint16)
    slice2 = (myarray[1] * 65535).astype(np.uint16)
    slice3 = (myarray[2] * 65535).astype(np.uint16)
    slice4 =  (np.amax(myarray, axis=0) * 65535).astype(np.uint16)

    cv2.imwrite('cv2_img_bg.png', slice1)
    cv2.imwrite('cv2_img_lumen.png', slice2)
    cv2.imwrite('cv2_img_media.png', slice3)
    cv2.imwrite('cv2_img_combined.png', slice4)
'''


for key, myarray in myarray_obj.items():
    #print(np.min(myarray), np.max(myarray)) #0.0, 1.0

    myarray = np.squeeze(myarray) # 3,384,384
    myarray = (myarray * 65535).astype(np.uint16)
    myarray = np.moveaxis(myarray, 0, -1) # 384,384,3
    cv2.imwrite('cv2_img_color.png', myarray)