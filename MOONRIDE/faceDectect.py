from aip import AipFace
import base64
import cv2
import urllib.parse

APP_ID = '11473153'
API_KEY = 'If9GTyNGXfyb6GmMjRb48OpX'
SECRET_KEY = '10AKQg5l5eT5Imf2HzZB4CGsRuXeq7kp'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

f = open('tp.jpg','rb') #二进制方式打开图文件
ls_f=base64.b64encode(f.read()) #读取文件内容，转换为base64编码
f.close()
s=urllib.parse.quote(ls_f)
print(s)
imageType = "BASE64"

options = {}
options["face_field"] = "age,beauty,expression,faceshape,gender,glasses,landmark,race"
options["max_face_num"] = 1
options["face_type"] = "LIVE"

result = client.detect(s, imageType, options)

print(result)
