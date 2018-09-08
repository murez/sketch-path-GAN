import cv2
import numpy as np
import math
import os


def rotate_img(img, angle):
    row, col = img.shape
    M = cv2.getRotationMatrix2D((row / 2, col / 2), angle, 1)
    res = cv2.warpAffine(img, M, (row, col))
    return res


def get_eight_directions(l_len):
    L = np.zeros((8, l_len, l_len))
    half_len = (l_len + 1) / 2
    for i in range(8):
        if i == 0 or i == 1 or i == 2 or i == 7:
            for x in range(l_len):
                y = int(half_len - int(round((x + 1 - half_len) * math.tan(math.pi * i / 8))))
                if y > 0 and y <= l_len:
                    L[i, x, y - 1] = 1
            if i != 7:
                L[i + 4] = np.rot90(L[i])
    L[3] = np.rot90(L[7], 3)
    return L


# compute and get the stroke of the raw img
def get_stroke(img, ks, dirNum):
    height, width = img.shape[0], img.shape[1]
    img = np.float32(img) / 255.0
    img = cv2.medianBlur(img, 3)
    imX = np.append(np.absolute(img[:, 0: width - 1] - img[:, 1: width]), np.zeros((height, 1)), axis=1)
    imY = np.append(np.absolute(img[0: height - 1, :] - img[1: height, :]), np.zeros((1, width)), axis=0)
    img_gredient = imX + imY

    kernel_Ref = np.zeros((ks * 2 + 1, ks * 2 + 1))
    kernel_Ref[ks, :] = 1

    response = np.zeros((dirNum, height, width))
    L = get_eight_directions(2 * ks + 1)
    for n in range(dirNum):
        ker = rotate_img(kernel_Ref, n * 180 / dirNum)
        response[n, :, :] = cv2.filter2D(img, -1, ker)

    Cs = np.zeros((dirNum, height, width))
    for x in range(width):
        for y in range(height):
            i = np.argmax(response[:, y, x])
            Cs[i, y, x] = img_gredient[y, x]

    spn = np.zeros((8, img.shape[0], img.shape[1]))

    kernel_Ref = np.zeros((2 * ks + 1, 2 * ks + 1))
    kernel_Ref[ks, :] = 1
    for n in range(width):
        if (ks - n) >= 0:
            kernel_Ref[ks - n, :] = 1
        if (ks + n) < ks * 2:
            kernel_Ref[ks + n, :] = 1

    kernel_Ref = np.zeros((2 * ks + 1, 2 * ks + 1))
    kernel_Ref[ks, :] = 1

    for i in range(8):
        ker = rotate_img(kernel_Ref, i * 180 / dirNum)
        spn[i] = cv2.filter2D(Cs[i], -1, ker)

    sp = np.sum(spn, axis=0)
    sp = (sp - np.min(sp)) / (np.max(sp) - np.min(sp))
    S = 1 - sp
    S = S * 255
    return S


filepath = "D:/pic/CNHK/photo/"
files = os.listdir(filepath)
for file in files:
    if not os.path.isdir(file):
        img = cv2.imread("D:/pic/CNHK/photo/"+file, 0)
        p=get_stroke(img,3,8)
        image = np.zeros((768,768),np.uint8)
        image = p[0:768,128:1024-128]
        cv2.imwrite('D:/pic/source/'+'pertreated'+file,image)
'''
filepath = 'f1-001-01.jpg'

S = get_stroke(simg, 3, 8)
S = S * 225
cv2.imwrite('pertreated.jpg', S)
'''