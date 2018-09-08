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


filepath = "D:/pic/CNHK/photo/"
files = os.listdir(filepath)
for file in files:
    if not os.path.isdir(file):
        path1 = filepath + file
        file = file.rstrip('.jpg')
        file = file.rstrip('.JPG')
        file = file.rstrip('.jpeg')
        file = file.rstrip('.JPEG')
        path2 = 'D:/pic/CNHK/sketch/' + file + '-sz1.jpg'
        img1 = cv2.imread(path1, 0)
        img2o = cv2.imread(path2, 0)
        sketchhead = img2o[82:img2o.shape[0], 3:img2o.shape[1]-3]
        img2 = np.zeros((768,768),np.uint8)
        img2.fill(255)
        img2[0:768, 66:702] = cv2.resize(sketchhead, (636, 768))
        img2 = cv2.medianBlur(img2,3)
        cv2.imwrite('D:/pic/aim' + 'pertreated' + file+'.jpg', img2)