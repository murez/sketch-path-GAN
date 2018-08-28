import requests
from json import JSONDecoder
import datetime
import cv2

def detect_face(http_url,key,secret,filepath1):
    data = {"api_key":key, "api_secret": secret, "return_landmark": "1"}
    files = {"image_file": open(filepath1, "rb")}
    img=cv2.imread(filepath1)
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
    faces=req_dict['faces']
    for i in range(len(faces)):
        face_rectangle = faces[i]['face_rectangle']
        width = face_rectangle['width']
        top = face_rectangle['top']
        left = face_rectangle['left']
        height = face_rectangle['height']
        start = (left, top)
        end = (left + width, top + height)
        color = (55, 255, 155)
        thickness = 3
        cv2.rectangle(img, start, end, color, thickness)
    cv2.imshow("a",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

http_url ="https://api-cn.faceplusplus.com/facepp/v3/detect"
key ="s7iWsJnl0ZfAMJu_IZ4V5mnZyinMGz0n"
secret ="o6USx6dPtPKrC_hTO-znQn4WV1zZbyEF"
filepath1 ="f1-001-01.jpg"
detect_face(http_url,key,secret,filepath1)