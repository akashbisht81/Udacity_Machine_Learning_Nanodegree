import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


df = pd.read_table('smsspamcollection/SMSSpamCollection',
                    sep = '\t',
                    header = None,
                    names = ['label','sms_message'])

df['label'] = df.label.map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state = 1)

count_vector = CountVectorizer()

training_data = count_vector.fit_transform(X_train)

testing_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()

naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score', format(precision_score(y_test, predictions)))
print('Recall score:', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
