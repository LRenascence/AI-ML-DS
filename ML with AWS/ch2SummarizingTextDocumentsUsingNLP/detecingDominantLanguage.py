import boto3
import json

comprehend = boto3.client(service_name = 'comprehend', region_name = 'us-east-1')

english_string = 'Machine Learning is fascinating.'
spanish_string = 'El aprendizaje automático es fascinante.'
chinese_string = '机器学习'

print('Calling DetectDominantLanguage')
print('english_string result: ')

print(json.dumps(comprehend.detect_dominant_language(Text = english_string), sort_keys = True, indent = 4))

print('\nspanish_string_list results:')
print(json.dumps(comprehend.detect_dominant_language(Text = spanish_string), sort_keys = True, indent = 4))

print('\nchinese_string_list results:')
print(json.dumps(comprehend.detect_dominant_language(Text = chinese_string), sort_keys = True, indent = 4))