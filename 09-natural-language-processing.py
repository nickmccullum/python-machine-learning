#Import the stopwords identifier
import nltk
nltk.download_shell()

#Import the data set
data = [line.rstrip() for line in open('SMSSpamCollection')]

#Run our data imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Create our DataFrame
data_frame = pd.read_csv('SMSSpamCollection', sep = '\t', names = ['type', 'message'])

#Exploratory Data Analysis
data_frame.describe()
data_frame.groupby('type').describe()
data_frame['message length'] = data_frame['message'].apply(len)
sns.distplot(data_frame['message length'])
data_frame.hist(column='message length', by='type', figsize=(13,5))

#Text preprocessing
import string
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')

#Example of text preprocessing
sample_message = 'This is a sample message! It has punctuation... will we be able to remove it?'
message_without_punctuation = ''.join([char for char in sample_message if char not in string.punctuation])
cleaned_message = ' '.join([word for word in message_without_punctuation.split() if word.lower() not in stopwords_list])

#Building a text preprocessing function
def preprocessor(message):
    """
    This function accepts a SMS message and performs two main actions:
    1. Removes punctuation from the SMS message
    2. Removes stop words (defined by the nltk library) from the SMS message
    The function returns a Python list.
    """
    
    message_without_punctuation = ''.join([char for char in message if char not in string.punctuation])
    return [word for word in message_without_punctuation.split(' ') if word.lower() not in stopwords.words('english')]

#Testing the function
preprocessor(sample_message)

#Tokenizing the data set
# data_frame['message'] = data_frame['message'].apply(preprocessor)

#Vectorizing the data set
from sklearn.feature_extraction.text import CountVectorizer
bag_of_words_builder = CountVectorizer(analyzer = preprocessor).fit(data_frame['message'])
len(bag_of_words_builder.vocabulary_)

#Testing our bag of words transformation
first_message = data_frame['message'][0]
# print(first_message)

first_bag_of_words = bag_of_words_builder.transform([first_message])
# print(first_bag_of_words)
# print(bag_of_words_builder.get_feature_names()[11165])

#Creating a bag of words matrix
bag_of_words_matrix = bag_of_words_builder.transform(data_frame['message'])

#Importing the TD-IDF class
from sklearn.feature_extraction.text import TfidfTransformer

#Calculating a TF-IDF value
tfidf_builder = TfidfTransformer().fit(bag_of_words_matrix)
first_message_tfidf = tfidf_builder.transform(first_bag_of_words)
# print(first_message_tfidf)

#Building the TF-IDF matrix
tfidf_matrix = tfidf_builder.transform(bag_of_words_matrix)

#Import the multinomial naive bayes theorem class
from sklearn.naive_bayes import MultinomialNB

#Training the model
spam_detector = MultinomialNB().fit(tfidf_matrix, data_frame['type'])

#Making predictions
spam_detector.predict(first_message_tfidf)[0]

#Splitting our data into training data and test data
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(data_frame['message'], data_frame['type'], test_size = 0.3)

#Build our data pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('create_bow', CountVectorizer(analyzer=preprocessor)),
    ('calculate_tfidf', TfidfTransformer()),
    ('make_prediction', MultinomialNB())
])

#Fit the pipeline and make predictions
pipeline.fit(x_training_data, y_training_data)
predictions = pipeline.predict(x_test_data)

#Measure the performance of our natural language processing algorithm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

classification_report(y_test_data, predictions)
confusion_matrix(y_test_data, predictions)
