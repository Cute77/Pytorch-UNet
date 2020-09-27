from os import listdir
import shutil
import random

# intputpath = 'ISIC-2017_Training_Data'
# outputpath = 'ISIC-2017_Training_Data_validation'
# dirname = 'ISIC-2017_Training_Part1_GroundTruth_validation'
# inputpath = 'ISIC-2017_Training_Data_validation'
dirname = 'ISIC-2017_Training_Data_clean'
inputpath = 'ISIC-2017_Training_Part1_GroundTruth_validation'
outputpath = 'ISIC-2017_Training_Part1_GroundTruth_validation_clean'



name = []

for filename in listdir(dirname):
    line = 'ISIC_' + filename.split('_')[1] + '_segmentation.png'
    name.append(line)

# namefinal = name[:5]

# random.shuffle(name)
# offset = int(len(name) * 0.1)
# data = name[:offset]

for line in name:
    src = inputpath + '/' + line
    print(src)
    dst = outputpath + '/' + line
    print(dst)
    shutil.move(src, dst)



