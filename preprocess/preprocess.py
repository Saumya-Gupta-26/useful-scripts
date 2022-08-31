import glob, os, sys
import argparse
import json

import SimpleITK as sitk

def main(args):

    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)


    ### Creating list of input volume filenames
    if len(params["location"]["input_file"]) != 0:
        input_file_list = params["location"]["input_file"]
    
    elif len(params["location"]["input_folder"]) != 0:
        input_file_list = []
        for i in range(len(params["location"]["input_folder"])):
            temp_list = glob.glob(str(params["location"]["input_folder"][i]) + "*" + params["location"]["extension"])
            for f in temp_list:
                input_file_list.append(f)
            del temp_list
    
    else:
        print("Please fix \"location\" parameters in json file.")
        sys.exit(1)

    number_input_files = len(input_file_list)
    output_folder_path = str(params["location"]["output_folder"])
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    print("Total number of input files is {}".format(number_input_files))

    for fi in range(number_input_files):
        sitkimage_gt    = sitk.ReadImage(input_file_list[fi])
        suffix = ""

        ### Remove small branches from segmentation
        if len(params["apply"]["remove_branch"]) == 1:
            labelToPrune = params["apply"]["remove_branch"][0]
            suffix += "_mainBranch"

            sitkimage_gt_binary    = sitkimage_gt

            ## Set all labels except background to 1
            correspondingImageFile = input_file_list[fi].split('/')[-1]
            correspondingImageFile = "/data/aorta/original/" + correspondingImageFile
            tempDiv = correspondingImageFile.split('.')
            correspondingImageFile = tempDiv[0] + '.' + tempDiv[1][0] + params["location"]["extension"]
            sitkimage       = sitk.ReadImage(correspondingImageFile)
            ### Get label information
            label_info = sitk.LabelStatisticsImageFilter()
            label_info.Execute(sitkimage, sitkimage_gt)
            current_labels = label_info.GetLabels()

            labelDict = {0:0, 1:1, 2:1} # combining green and red into red
            changeLabel = sitk.ChangeLabelImageFilter()
            changeLabel.SetChangeMap(labelDict)
            sitkimage_gt_binary = changeLabel.Execute(sitkimage_gt_binary)

            ### temp code ### delete 
            #output_filename = input_file_list[fi].rsplit('.',1)[0] + "_onlyONE" + '.' + input_file_list[fi].rsplit('.',1)[1]
            #output_filename = os.path.join(output_folder_path, output_filename.rsplit('/',1)[1])
            #sitk.WriteImage(sitkimage_gt, output_filename)

            ## Apply Opening OR Erosion,Dilation
            removeStray = sitk.BinaryMorphologicalOpeningImageFilter()
            #removeStray.SetKernelType(sitk.sitkBall)
            removeStray.SetKernelRadius(1)
            removeStray.SetForegroundValue(labelToPrune)
            sitkimage_gt_binary = removeStray.Execute(sitkimage_gt_binary)

            ## Use CC to get largest component of aorta main branch
            # Assign label to each CC
            allCC = sitk.ConnectedComponentImageFilter()
            sitkimage_gt_binary = allCC.Execute(sitkimage_gt_binary)
            # Find largest CC
            largestCC = sitk.LabelShapeStatisticsImageFilter()
            largestCC.Execute(sitkimage_gt_binary)
            allLabels = largestCC.GetLabels()
            maxNum = 0
            maxLabel = 0
            for alabel in allLabels:
                curNum = largestCC.GetNumberOfPixels(alabel)
                if curNum > maxNum:
                    maxNum = curNum
                    maxLabel = alabel
            # Assign non-largest CC to background 0
            ccDict = {}
            for alabel in allLabels:
                if alabel != maxLabel:
                    ccDict[alabel] = 0
                else:
                    ccDict[alabel] = labelToPrune
            changeLabel = sitk.ChangeLabelImageFilter()
            changeLabel.SetChangeMap(ccDict)
            sitkimage_gt_binary = changeLabel.Execute(sitkimage_gt_binary)

            # We have the largest CC in _binary. We now need to separate it into red and green.
            arrayimage_gt = sitk.GetArrayFromImage(sitkimage_gt)
            arrayimage_gt_binary = sitk.GetArrayFromImage(sitkimage_gt_binary)
            gt_size = arrayimage_gt.shape # 263,512,512

            for z in range(gt_size[0]):
                for y in range(gt_size[1]):
                    for x in range(gt_size[2]):
                        if arrayimage_gt_binary[z][y][x] == 1:
                            if arrayimage_gt[z][y][x] == 2:
                                arrayimage_gt_binary[z][y][x] = 2

            modified_sitkimage_gt = sitk.GetImageFromArray(arrayimage_gt_binary)
            modified_sitkimage_gt.SetSpacing(sitkimage_gt.GetSpacing())
            modified_sitkimage_gt.SetOrigin(sitkimage_gt.GetOrigin())
            modified_sitkimage_gt.SetDirection(sitkimage_gt.GetDirection())
            
            del sitkimage_gt, arrayimage_gt, arrayimage_gt_binary
            sitkimage_gt = modified_sitkimage_gt
            del modified_sitkimage_gt

        else:
            print("No branches to remove OR cannot handle more than one label at a time")



        ### Change labels
        if len(params["apply"]["change_labels"]) != 0:
            correspondingImageFile = input_file_list[fi].split('/')[-1]
            correspondingImageFile = "/data/aorta/original/" + correspondingImageFile
            tempDiv = correspondingImageFile.split('.')
            correspondingImageFile = tempDiv[0] + '.' + tempDiv[1][0] + params["location"]["extension"]
            sitkimage       = sitk.ReadImage(correspondingImageFile)
            suffix += "_correctedLabels"
            ### Get label information
            label_info = sitk.LabelStatisticsImageFilter()
            label_info.Execute(sitkimage, sitkimage_gt)
            original_labels = label_info.GetLabels()
            del sitkimage
            
            # make a dict # change all labels to temp values not in list # then change one-by-one to destination label value # this overcome cases where we need to reverse the labels
            originalLabelDict = {}
            tempLabelDict = {}
            finalLabelDict = {}

            for li in range(len(params["apply"]["change_labels"])):
                originalLabelDict[params["apply"]["change_labels"][li][0]] = params["apply"]["change_labels"][li][1]

            maxLabel = max(original_labels) + 1
            for key,_ in originalLabelDict.items():
                tempLabelDict[key] = maxLabel
                maxLabel += 1
            
            changeLabel = sitk.ChangeLabelImageFilter()
            changeLabel.SetChangeMap(tempLabelDict)
            sitkimage_gt = changeLabel.Execute(sitkimage_gt)

            for key,value in tempLabelDict.items():
                finalLabelDict[value] = originalLabelDict[key]

            changeLabel = sitk.ChangeLabelImageFilter()
            changeLabel.SetChangeMap(finalLabelDict)
            sitkimage_gt = changeLabel.Execute(sitkimage_gt) 

            del tempLabelDict, originalLabelDict, finalLabelDict

        else:
            print("No change in labels")

        ### Fill holes
        if len(params["apply"]["fill_holes"]) != 0:
            suffix += "_closedHoles"
            for labelToFill in params["apply"]["fill_holes"]:
                fillHole = sitk.BinaryMorphologicalClosingImageFilter()
                #fillHole.SetKernelType(sitk.sitkBall)
                #fillHole.SetKernelRadius(1)
                fillHole.SetForegroundValue(labelToFill)
                sitkimage_gt = fillHole.Execute(sitkimage_gt)
                # default values: kernelRadius = std::vector<uint32_t>(3, 1), KernelEnum kernelType = itk::simple::sitkBall, double foregroundValue = 1.0, bool safeBorder = true );

        else:
            print("No label holes to fill")

        ### Add minimum thickness boundary
        if len(params["apply"]["add_minimum_label_thickness"]) != 0:
            suffix += "_wallBoundary"
            for eleList in params["apply"]["add_minimum_label_thickness"]:
                centerLabel = eleList[0]
                srcLabel = 0
                destLabel = eleList[1]
                thickness = [eleList[2],eleList[3],eleList[4]]

                arrayimage_gt = sitk.GetArrayFromImage(sitkimage_gt)
                gt_size = arrayimage_gt.shape # 263,512,512

                for z in range(gt_size[0]):
                    for y in range(gt_size[1]):
                        for x in range(gt_size[2]):
                            if arrayimage_gt[z][y][x] == centerLabel:
                                # Get Neighbours
                                    for nz in range(z-thickness[0], z+thickness[0]+1): # +1 because range doesn't include upperlimit
                                        for ny in range(y-thickness[1], y+thickness[1]+1):
                                            for nx in range(x-thickness[2], x+thickness[2]+1):
                                                if checkBounds([nz,ny,nx],gt_size) and arrayimage_gt[nz][ny][nx]==srcLabel:
                                                    arrayimage_gt[nz][ny][nx] = destLabel

                modified_sitkimage_gt = sitk.GetImageFromArray(arrayimage_gt)
                modified_sitkimage_gt.SetSpacing(sitkimage_gt.GetSpacing())
                modified_sitkimage_gt.SetOrigin(sitkimage_gt.GetOrigin())
                modified_sitkimage_gt.SetDirection(sitkimage_gt.GetDirection())
                
                del sitkimage_gt, arrayimage_gt
                sitkimage_gt = modified_sitkimage_gt
                del modified_sitkimage_gt

        else:
            print("No label thickness to be added")

        ### Save volume to file
        output_filename = input_file_list[fi].rsplit('.',1)[0] + suffix + '.' + input_file_list[fi].rsplit('.',1)[1]
        output_filename = os.path.join(output_folder_path, output_filename.rsplit('/',1)[1])
        print(output_filename)
        sitk.WriteImage(sitkimage_gt, output_filename)


def checkBounds(curPos, volSize):
    if curPos[0] < 0 or curPos[0] >= volSize[0]:
        return 0
    if curPos[1] < 0 or curPos[1] >= volSize[1]:
        return 0
    if curPos[2] < 0 or curPos[2] >= volSize[2]:
        return 0
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        main(args)


