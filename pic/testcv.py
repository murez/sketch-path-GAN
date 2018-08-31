import requests
from json import JSONDecoder
import datetime
import cv2
import numpy as np
import os

http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
key = "s7iWsJnl0ZfAMJu_IZ4V5mnZyinMGz0n"
secret = "o6USx6dPtPKrC_hTO-znQn4WV1zZbyEF"


def detect_face(http_url, key, secret, filepath1):
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
    return req_dict


def analyze_response(req_dict):
    faces = req_dict['faces']
    for i in range(len(faces)):
        face_rectangle = faces[i]['face_rectangle']
        left = face_rectangle['left']
        width = face_rectangle['width']
        top = face_rectangle['top']
        height = face_rectangle['height']
    return left, width, top, height


path1 = './CNHK/photo/f1-001-01.jpg'
path2 = './CNHK/sketch/f1-001-01-sz1.jpg'
img1 = cv2.imread(path1, 0)
img2o = cv2.imread(path2, 0)
response1 = detect_face(http_url, key, secret, path1)
left1, width1, top1, height1 = analyze_response(response1)
response2 = detect_face(http_url, key, secret, path2)
left2, width2, top2, height2 = analyze_response(response2)
img2 = np.zeros((768, 1024, 1), np.uint8)
rate = float(width1 / width2)
sketchhead = img2o[82:img2o.shape[0], 0:img2o.shape[1]]
sketchhead = cv2.resize(sketchhead, (int(sketchhead.shape[0] * rate), int(sketchhead.shape[1] * rate)))
sketchhead = cv2.medianBlur(sketchhead, 3)
img2[img2.shape[0] - sketchhead.shap[0]:img2.shape[0],
int(img2.shape[1] / 2 - sketchhead.shape[1] / 2):int(img2.shape[1] / 2 + sketchhead.shape[1] / 2)] = sketchhead
cv2.imwrite('resize.jpg', img2)