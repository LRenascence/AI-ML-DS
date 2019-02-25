import boto3
import pandas as pd

s3 = boto3.client('s3')

bucket_name = 'renascence-known-tm-analysis'
s3.create_bucket(Bucket = bucket_name)

filenames_list = ['doc-topics.csv', 'topic-terms.csv']
for filename in filenames_list:
    s3.upload_file(filename, bucket_name, filename)
    if filename == 'doc-topics.csv':
        obj = s3.get_object(Bucket = bucket_name, Key = filename)
        doc_topics = pd.read_csv(obj['Body'])
    else:
        obj = s3.get_object(Bucket = bucket_name, Key = filename)
        topic_terms = pd.read_csv(obj['Body'])

merged_df = pd.merge(doc_topics, topic_terms, on = 'topic')
print(merged_df)