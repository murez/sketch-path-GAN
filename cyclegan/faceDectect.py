import requests
from json import JSONDecoder
import datetime
import cv2


def detect_face(http_url, key, secret, filepath1):
    data = {"api_key": key, "api_secret": secret, "return_landmark": "1"}
    files = {"image_file": open(filepath1, "rb")}
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

def analyze_response(req_dict, img):
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
        nose_right_x = landmark['nose_right']['x']
        nose_left_x = landmark['nose_left']['x']
        nose_lowermiddle_y = landmark['nose_contour_lower_middle']['y']
        nose_rightup_y = landmark['nose_contour_right1']['y']
        left_eye_leftcorner_x = landmark['left_eye_left_corner']['x']
        left_eye_top_y = landmark['left_eye_top']['y']
        left_eye_rightcorner_x = landmark['left_eye_right_corner']['x']
        left_eye_bottom_y = landmark['left_eye_bottom']['y']
        right_eye_leftcorner_x = landmark['right_eye_left_corner']['x']
        right_eye_top_y = landmark['right_eye_top']['y']
        right_eye_rightcorner_x = landmark['right_eye_right_corner']['x']
        right_eye_bottom_y = landmark['right_eye_bottom']['y']
        left_eyebrow_leftcorner_x = landmark['left_eyebrow_left_corner']['x']
        left_eyebrow_lower_middle_y = landmark['left_eyebrow_lower_middle']['y']
        left_eyebrow_rightcorner_x = landmark['left_eyebrow_right_corner']['x']
        left_eyebrow_upper_middle_y = landmark['left_eyebrow_upper_middle']['y']
        right_eyebrow_leftcorner_x = landmark['right_eyebrow_left_corner']['x']
        right_eyebrow_lower_middle_y = landmark['right_eyebrow_lower_middle']['y']
        right_eyebrow_rightcorner_x = landmark['right_eyebrow_right_corner']['x']
        right_eyebrow_upper_middle_y = landmark['right_eyebrow_upper_middle']['y']
        color = (55, 255, 155)
        thickness = 3
        cv2.rectangle(img, (mouth_leftup_x, mouth_leftup_y), (mouth_rightdown_x, mouth_rightdown_y), color, thickness)
        cv2.rectangle(img, (nose_left_x, nose_lowermiddle_y), (nose_right_x, nose_rightup_y), color, thickness)
        cv2.rectangle(img, (left_eye_leftcorner_x, left_eye_top_y), (left_eye_rightcorner_x, left_eye_bottom_y), color,
                      thickness)
        cv2.rectangle(img, (right_eye_leftcorner_x, right_eye_top_y), (right_eye_rightcorner_x, right_eye_bottom_y),
                      color, thickness)
        cv2.rectangle(img, (left_eyebrow_leftcorner_x, left_eyebrow_lower_middle_y),
                      (left_eyebrow_rightcorner_x, left_eyebrow_upper_middle_y), color, thickness)
        cv2.rectangle(img, (right_eyebrow_leftcorner_x, right_eyebrow_lower_middle_y),
                      (right_eyebrow_rightcorner_x, right_eyebrow_upper_middle_y), color, thickness)
    cv2.imshow("mouth", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
key = "s7iWsJnl0ZfAMJu_IZ4V5mnZyinMGz0n"
secret = "o6USx6dPtPKrC_hTO-znQn4WV1zZbyEF"
filepath1 = "f1-001-01.jpg"
img = cv2.imread(filepath1,0)
response = detect_face(http_url, key, secret, filepath1)
analyze_response(response, img)