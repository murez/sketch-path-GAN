import cv2
import os
import numpy as np

filepath = "G:/Python-Projects/pic/CNHK/photo/"
files = os.listdir(filepath)
for file in files:
    if not os.path.isdir(file):
        img = cv2.imread(filepath + file,1)
        image = np.zeros((768, 768), np.uint8)
        image = img[0:768, 128:(1024-128)]
        cv2.imwrite(filepath + 'source/' + 'pertreated' + file, image)
