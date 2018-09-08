import os,base64,cv2
import requests
import numpy as np
from json import JSONDecoder

http_url = "https://api-cn.faceplusplus.com/humanbodypp/v2/segment"
key = "s7iWsJnl0ZfAMJu_IZ4V5mnZyinMGz0n"
secret = "o6USx6dPtPKrC_hTO-znQn4WV1zZbyEF"
fp_source = "G:/Python-Projects/pic/CNHK/photo/source/"
files = os.listdir(fp_source)

for file in files:
    if not os.path.isdir(file):
        data = {"api_key": key, "api_secret": secret}
        files = {"image_file": open(fp_source + file, "rb")}
        response = requests.post(http_url, data=data, files=files)
        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)
        body_img = req_dict['body_image']
        dec_img = base64.b64decode(body_img)
        fp = open('G:/Python-Projects/pic/source/' + 'segmented-' + file, 'wb')
        fp.write(dec_img)
        fp.close()