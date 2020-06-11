#Numerical computing libraries
import pandas as pd
import numpy as np

#Visalization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

raw_data = pd.read_csv('kyphosis-data.csv')
raw_data.columns

#Exploratory data analysis
raw_data.info()
sns.pairplot(raw_data, hue = 'Kyphosis')

#Split the data set into training data and test data
from sklearn.model_selection import train_test_split
x = raw_data.drop('Kyphosis', axis = 1)
y = raw_data['Kyphosis']
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

#Train the decision tree model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_training_data, y_training_data)
predictions = model.predict(x_test_data)

#Measure the performance of the decision tree model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(y_test_data, predictions))
print(confusion_matrix(y_test_data, predictions))

#Train the random forests model
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier()
random_forest_model.fit(x_training_data, y_training_data)
random_forest_predictions = random_forest_model.predict(x_test_data)

#Measure the performance of the random forest model
print(classification_report(y_test_data, random_forest_predictions))
print(confusion_matrix(y_test_data, random_forest_predictions))
