import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

subscription_key = '0193998749b44c7ba01fce150e9677f9'
vision_base_url = 'https://albuser10computervision.cognitiveservices.azure.com/vision/v2.0/'
ocr_url = vision_base_url + 'ocr'

image_url = 'https://i.stack.imgur.com/WiDpa.jpg'

image = Image.open(BytesIO(requests.get(image_url).content))
image

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
params = {'language': 'unk', 'detectOrientation': 'true'}
data = {'url':image_url}

response = requests.post(ocr_url, headers=headers, params=params, json=data)

result = response.json()
result

for region in result['regions']:
  lines = region['lines']

  for line in lines:
    words = line['words']

    for word in words:
      print(word['text'])
      
image_url = "https://www.unikorea.go.kr/unikorea/common/images/content/peace.png"
image = Image.open(BytesIO(requests.get(image_url).content))
image

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
params = {'language': 'ko', 'detectOrientation': 'true'}
data = {'url':image_url}

response = requests.post(ocr_url, headers=headers, params=params, json=data)
result = response.json()
result

for region in result['regions']:
  lines = region['lines']

  for line in lines:
    words = line['words']

    for word in words:
      print(word['text'])

subscription_key