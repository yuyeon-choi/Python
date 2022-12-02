import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

subscription_key = '04d93910ed9d4ee7a6a33dd282ac07f0'
face_api_url = 'https://labuser10face.cognitiveservices.azure.com/face/v1.0/detect'

#Class, library, Package 대문자 관례
#지역변수, 파라메타 소문자로 관례
#addr, msg 줄임말은 배제
#두 단어가 합쳐지면 두 번째 단어는 대문자
#상수는 전체가 대문자 const MAX_USER = 100

image_url = 'http://image.koreatimes.com/article/2021/05/10/20210510094734601.jpg'

image = Image.open(BytesIO(requests.get(image_url).content))

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'Smile'
}

data = {'url': image_url}

response = requests.post(face_api_url, params=params, headers=headers,json=data)
faces = response.json()
faces

draw = ImageDraw.Draw(image)

def DrawBox(faces):

  for face in faces:
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    width = rect['width']
    height = rect['height']

    draw.rectangle(((left,top),(left+width,top+height)),outline='red')

    face_attributes = face['faceAttributes']
    smile = face_attributes['smile']
    draw.text((left, top),str(smile), fill='red')

# DrawBox(faces)
# image