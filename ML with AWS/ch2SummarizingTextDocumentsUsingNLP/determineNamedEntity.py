import boto3
import json

comprehend = boto3.client(service_name = 'comprehend', region_name = 'us-east-1')

english_string = 'I study Machine Learning in Dallas today'

print('Calling DetectEntites')

print(json.dumps(comprehend.detect_entities(Text = english_string, LanguageCode = 'en'), sort_keys = True, indent = 4))
