from os import listdir
import shutil
import random

dirname = 'ISIC-2017_Training_Part1_GroundTruth'
path = 'ISIC-2017_Training_Part1_GroundTruth_validation'
name = []

for filename in listdir(dirname):
    name.append(filename)

random.shuffle(name)

offset = len(name) * 0.1
data = name[:offset]

for line in data:
    src = dirname + '/' + line
    dst = path + '/' + line
    shutil.move(src, dst)
    


