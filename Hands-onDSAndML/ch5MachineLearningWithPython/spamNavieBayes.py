import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# read the data from csv file
df = pd.read_csv('spam.csv', encoding = "ISO-8859-1")


train = df[:5000]
test = df[5001:]



# counts is the word list and number of times
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(train.v2)
# target is spam or ham
classifier = MultinomialNB()
targets = train.v1
classifier.fit(counts, targets)

# run on test set
test_counts = vectorizer.transform(test.v2)
predictions = classifier.predict(test_counts)


match = 0
result = test['v1'].values

for i in range(len(predictions)):
    if predictions[i] == result[i]:
        match += 1

print(match)
print(match / len(predictions))
