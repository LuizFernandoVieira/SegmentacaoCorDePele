import cv2
import numpy as np
import operator
from os import listdir
from os.path import isfile, join, isdir
directories = ['SFA']
while (len(directories) > 0):
    prefix = directories.pop()
    files = [f for f in listdir(prefix) if isfile(join(prefix, f))]
    bw_files = [f for f in bw_files if f != "243.jpg" and f != "278.jpg" and f != ".DS_Store"]
    gt_files = [f for f in gt_files if f != "243.jpg" and f != "278.jpg" if f != ".DS_Store"]
    print(files)
