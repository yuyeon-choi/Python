# !pip install azure_cognitiveservices-vision-customvision

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

ENDPOINT = 'https://labuser10custom.cognitiveservices.azure.com/'

training_key = '3777fe4bf1104f4d9a89122f1f9bb4c3'
prediction_key = 'dd866206b3864dad9f63629fd5b72198'
prediction_resource_id = '/subscriptions/7ae06d59-97e1-4a36-bbfe-efb081b9b03b/resourceGroups/RG10/providers/Microsoft.CognitiveServices/accounts/labuser10custom'

publish_iteration_name = "classifyModel"

credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# https://github.com/Azure-Samples/cognitive-services-quickstart-code/blob/master/python/CustomVision/ImageClassification/CustomVisionQuickstart.py
# https://colab.research.google.com/drive/1DWb7RT_OAdWyTfOz_ryQgSzLojf_vAW2?usp=sharing
print ("Creating project...")
project = trainer.create_project("Labuser10 Project")

Jajangmyeon_tag = trainer.create_tag(project.id, "Jajangmyeon")
Champon_tag = trainer.create_tag(project.id, "Champon")
Tangsuyug_tag = trainer.create_tag(project.id, "Tangsuyug")

import time

cprint('Training....')
iteration = trainer.train_project(project.id)
while (iteration.status != 'Completed'):
  iteration = trainer.get_iteration(project.id, iteration.id)
  print('Training status' + iteration.status)

  time.sleep(2)

print('Done!')