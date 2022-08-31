# given coordinates / contours, create a filled mask
from PIL import Image
import glob, os, sys
import numpy as np
import cv2

src_dir = "/data/saumgupta/miccai/dataset/IVUS/Test_Set/Data_set_B/LABELS_obs2_v2"
#src_dir = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/Data_set_B/LABELS_COLOUR"
dst_dir = "/data/saumgupta/miccai/dataset/IVUS/Test_Set/Data_set_B/png/LABELS_obs2_v2"

#lum_frame_01_0001_003.txt
#frame_01_0001_003.png
#frame_01_0001_003_gt.png
def generateBlankMasks():
    gtfiles = glob.glob(src_dir + '/lum_*.txt')
    print('todo: {}'.format(len(gtfiles)))

    imgroot = "/data/saumgupta/miccai/dataset/IVUS/Training_Set/Data_set_B/DCM"
    for fp in gtfiles:
        imgpath = os.path.join(imgroot, fp.split('/')[-1].replace('lum_', '').replace('.txt', '.png'))

        arrayimg = np.asarray(Image.open(imgpath))
        #print(arrayimg.shape) #(384,384)
        gtimg = Image.new('L', arrayimg.shape, color = (0))
        gtimg.save(os.path.join(dst_dir, imgpath.split('/')[-1].replace('.png','_gt.png')))

def coord2fill():
    txtfiles = glob.glob(src_dir + '/lum_*.txt')

    for lumfile in txtfiles:
        medfile = lumfile.replace('lum', 'med')
        lumContours = [np.around(np.loadtxt(lumfile, delimiter=',')).astype(np.int32)]
        medContours = [np.around(np.loadtxt(medfile, delimiter=',')).astype(np.int32)]

        imgpath = os.path.join(dst_dir, lumfile.split('/')[-1].replace('lum_', '').replace('.txt', '_gt.png'))
        arrayimg = cv2.imread(imgpath)
        #print(arrayimg.shape)
        cv2.drawContours(arrayimg, medContours, -1, color=(0, 0, 255), thickness=cv2.FILLED)
        cv2.drawContours(arrayimg, lumContours, -1, color=(0, 255, 0), thickness=cv2.FILLED)
        cv2.imwrite(imgpath.replace('.png', '_color.png'), arrayimg)


def rgb2label():
    rgbfiles = glob.glob(dst_dir + '/*_color.png')
    
    for rfile in rgbfiles:
        rgbimg = cv2.imread(rfile)
        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        label1 = 1
        label2 = 2
        shape = (rgbimg.shape[0],rgbimg.shape[1])
        img_singlechannel = np.zeros(shape)
        for x in range(shape[0]):
            for y in range(shape[1]):
                if compFunc(rgbimg[x][y],color1):
                    img_singlechannel[x][y] = label1
                elif compFunc(rgbimg[x][y],color2):
                    img_singlechannel[x][y] = label2
        cv2.imwrite(os.path.join(dst_dir,rfile.split('/')[-1].replace('_color.png','.png')), img_singlechannel)


def compFunc(in1, in2):
    if in1[0] == in2[0] and in1[1] == in2[1] and in1[2] == in2[2]:
        return True
    return False


if __name__ == "__main__":
    generateBlankMasks()
    coord2fill()
    rgb2label()