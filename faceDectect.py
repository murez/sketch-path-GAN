from aip import AipFace
import base64

APP_ID = '11473153'
API_KEY = 'If9GTyNGXfyb6GmMjRb48OpX'
SECRET_KEY = '10AKQg5l5eT5Imf2HzZB4CGsRuXeq7kp'

client = AipFace(APP_ID, API_KEY, SECRET_KEY)

with open("timg.jpg","rb") as f:
    base64_data = base64.b64encode(f.read())
    print(base64_data)
    image = str(base64_data)

imageType = "BASE64"

client.detect(image, imageType);

options = {}
options["face_field"] = "age"
options["max_face_num"] = 2
options["face_type"] = "LIVE"

result = client.detect(image, imageType, options)

print(result)