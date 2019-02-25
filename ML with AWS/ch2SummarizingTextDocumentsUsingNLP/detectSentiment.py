import boto3
import json

comprehend = boto3.client(service_name = 'comprehend', region_name = 'us-east-1')

english_string = 'Today is my birthday, I am so happy'

print('calling DetectSentiment')
print(json.dumps(comprehend.detect_sentiment(Text = english_string, LanguageCode = 'en'), sort_keys = True, indent = 4))
