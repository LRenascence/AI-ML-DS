import boto3
import json

comprehend = boto3.client(service_name = 'comprehend', region_name = 'us-east-1')

english_string_list = ['Machine Learning is fascinating.', 'Studying Artificial Intelligence is my passion.']
spanish_string_list = ['El aprendizaje automático es fascinante.', 'Estudiar Inteligencia Artificial es mi pasión.']

print('Calling DetectDominantLanguage')
print('english_string result: ')

print(json.dumps(comprehend.batch_detect_dominant_language(TextList = english_string_list), sort_keys = True, indent = 4))

print('\nspanish_string_list results:')
print(json.dumps(comprehend.batch_detect_dominant_language(TextList = spanish_string_list), sort_keys = True, indent = 4))
