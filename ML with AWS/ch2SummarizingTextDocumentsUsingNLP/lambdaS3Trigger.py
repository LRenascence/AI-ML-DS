import boto3
import json


def lambda_handler(event, context):
    # TODO implement
    s3 = boto3.client('s3')

    if event:
        bucket = "renascence-ml-s3-trigger"
        text_file_obj = event["Records"][0]
        filename = str(text_file_obj['s3']['object']['key'])

        print("filename: ", filename)

        file_obj = s3.get_object(Bucket=bucket, Key=filename)
        body_str_obj = str(file_obj['Body'].read())

        comprehend = boto3.client(service_name="comprehend", region_name='us-east-1')  # us-east-1
        sentiment_response = comprehend.detect_sentiment(Text=body_str_obj, LanguageCode="en")
        print("sentiment_response: \n", sentiment_response)

        entity_response = comprehend.detect_entities(Text=body_str_obj, LanguageCode="en")
        print("\n\nentity_response: \n", entity_response)

        key_phases_response = comprehend.detect_key_phrases(Text=body_str_obj, LanguageCode="en")
        print("\n\nkey_phases_response: \n", key_phases_response)

        return 'Hello from Lambda'
