from aip import AipImageClassify

APP_ID = '11383639'
API_KEY = 'Rj6emP5R8ZvEz2aRlU3mnn6l'
SECRET_KEY = 'wI9GqH5430Wa6ueqtkucRI9PQnsiqtwQ'

client = AipImageClassify(APP_ID,API_KEY,SECRET_KEY)
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('example2.gif')

print(client.advancedGeneral(image))
