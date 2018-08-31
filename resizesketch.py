import requests
from json import JSONDecoder
import datetime
import cv2

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
        width = face_rectangle['width']
        top = face_rectangle['top']
        left = face_rectangle['left']
        # height = face_rectangle['height']
    return width,top,left


filepath = "G:/Python-Projects/untitled/pic/CNHK/photo"
files = os.listdir(filepath)
for file in files:
    if not os.path.isdir(file):
        path1="G:/Python-Projects/untitled/pic/CNHK/photo/"+file
        path2="G:/Python-Projects/untitled/pic/CNHK/photo/"+file.rstrip('.jpg')+'-sz1.jpg'
        img1 = cv2.imread(path1, 0)
        img2 = cv2.imread(path2, 0)
        img = cv2.imread(filepath1,0)
        response1 = detect_face(http_url, key, secret, path1)
        analyze_response(response1)
        response2 = detect_face(http_url, key, secret, path2)
        analyze_response(response2)
