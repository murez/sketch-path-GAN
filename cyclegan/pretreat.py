import cv2
import numpy as np
import math
import requests
from json import JSONDecoder
import datetime


# face data get
def detect_face(filepath1):
    http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    key = "s7iWsJnl0ZfAMJu_IZ4V5mnZyinMGz0n",
    secret = "o6USx6dPtPKrC_hTO-znQn4WV1zZbyEF"
    data = {"api_key": key, "api_secret": secret, "return_landmark": "1"}
    files = {"image_file": open(filepath1, "rb")}
    img = cv2.imread(filepath1, 0)
    files = {"image_file": open(filepath1, "rb")}
    starttime = datetime.datetime.now()
    response = requests.post(http_url, data=data, files=files)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    req_con = response.content.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    fo = open("response", "w")
    fo.write(str(req_dict))
    fo.close()
    faces = req_dict['faces']
    for i in range(len(faces)):
        face_rectangle = faces[i]['face_rectangle']
        landmark = faces[i]['landmark']
        width = face_rectangle['width']
        top = face_rectangle['top']
        left = face_rectangle['left']
        height = face_rectangle['height']
        mouth_leftup_x = landmark['mouth_left_corner']['x']
        mouth_leftup_y = landmark['mouth_upper_lip_top']['y']
        mouth_rightdown_x = landmark['mouth_right_corner']['x']
        mouth_rightdown_y = landmark['mouth_lower_lip_bottom']['y']
        # cv2.rectangle(img, start, end, color, thickness)
    faceall = img[top:top + height, left:left + width]
    faceall = faceall
    lip = img[mouth_leftup_y:mouth_rightdown_y, mouth_leftup_x:mouth_rightdown_x]
    hair = img
    # cv2.imshow("hair", hair)
    # cv2.imshow("faceall", faceall)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return faceall, hair, lip


# 8 direction filter
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
    img = img * 0.8 + 0.2 * cv2.Laplacian(img, cv2.CV_16S, ksize=3)
    img = np.float32(img) / 255.0
    img = cv2.medianBlur(img, 3)
    # cv2.imshow('blur', img)
    imX = np.append(np.absolute(img[:, 0: width - 1] - img[:, 1: width]), np.zeros((height, 1)), axis=1)
    imY = np.append(np.absolute(img[0: height - 1, :] - img[1: height, :]), np.zeros((1, width)), axis=0)
    # img_gredient = np.sqrt((imX ** 2 + imY ** 2))
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
        # cv2.imshow('QAQ', spn[i])
        # cv2.waitKey(0)

    return spn


def pertreat(filepath):
    P1, P2 = detect_face(filepath)
    S1 = get_stroke(P1, 3, 8)
    S2 = get_stroke(P2, 3, 8)
    return S1, S2
