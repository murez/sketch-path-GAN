import os,cv2
import numpy as np

fp_source = 'G:/Python-Projects/pic/source/'
files = os.listdir(fp_source)

for file in files:
    if not os.path.isdir(file):
        img = cv2.imread(fp_source + file,)
        cv2.imwrite('G:/Python-Projects/pic/source/greyed/' + file, img)