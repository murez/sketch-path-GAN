import cv2
import numpy as np
import random

img = cv2.imread('/home/murez/PycharmProjects/untitled/pic/testimg.jpg',1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgG = cv2.GaussianBlur(gray,(3,3),0)
dst = cv2.Canny(img,12,180)
cv2.imwrite('000.jpg',dst)

imgfz = cv2.imread('000.jpg',1)
imgInfofz = imgfz.shape
dstfz = np.zeros((height,width,3),np.uint8)
for i in range(0,height):
    for j in range(0,width):
        (b,g,r) = imgfz[i,j]
        dstfz[i,j] = (255-b,255-g,255-r)
cv2.imwrite('000.jpg',dstfz)
cv2.imshow('',dstfz)
cv2.waitKey(0)