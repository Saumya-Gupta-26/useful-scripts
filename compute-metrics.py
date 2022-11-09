#compute metrics for 2D BINARY images
from PIL import Image
import numpy as np
import os, glob, sys
from skimage.metrics import hausdorff_distance, adapted_rand_error, variation_of_information

ddir = "/data/saumgupta/crf-dmt/DRIVE/unet/test-outputs/trial1-ce-dice"

def main():

    filelist = glob.glob(ddir+'/gt_*png')
    filelist.sort()

    avg = {'dice':0., 'hd':0., '1-ari':0., 'voi':0.}
    cnt = 0

    with open(os.path.join(ddir,'metric_values.txt'), 'w') as wfile:
        wfile.write("Folder: {}".format(ddir))
        for i, gtpath in enumerate(filelist):
            predpath = gtpath.replace("gt", "pred")

            if os.path.exists(predpath):
                cnt+=1

                pred = np.array(Image.open(predpath))/255.
                target = np.array(Image.open(gtpath))[:,:,0]/255. # saved as an RGB image

                diceCoeff = calculateDiceCoeff(pred, target)
                hdCoeff = calculateHDCoeff(pred,target)
                ariCoeff  = calculateARICoeff(pred,target)
                voiCoeff = calculateVOICoeff(pred,target)
                avg['dice'] += diceCoeff
                avg['hd'] += hdCoeff
                avg['1-ari'] += ariCoeff
                avg['voi'] += voiCoeff

                wfile.write("\n\nFile: {}\nDice: {}\nHD: {}\n1-ARI: {}\nVOI: {}".format(gtpath, diceCoeff, hdCoeff, ariCoeff, voiCoeff))

        wfile.write("\n\nFolder: {}\nTotal: {}\n".format(ddir,cnt))
        for key,item in avg.items():
            wfile.write("Avg {}: {}\n".format(key,item/cnt))


def calculateDiceCoeff(pred,target):

    m1 = pred.flatten().astype(np.float32)  # Flatten
    m2 = target.flatten().astype(np.float32)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection) / (m1.sum() + m2.sum())


def calculateHDCoeff(pred,target):
    return hausdorff_distance(pred, target)

def calculateARICoeff(pred,target):
    return 1.-adapted_rand_error(target.astype(np.int32),pred.astype(np.int32))[0]

def calculateVOICoeff(pred,target):
    return np.sum(variation_of_information(target.astype(np.int32),pred.astype(np.int32)))


if __name__ == "__main__":
    main()