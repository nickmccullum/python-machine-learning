#Data imports
import pandas as pd
import numpy as np

#Visualization imports
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Import the data set from scikit-learn
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
raw_data = pd.DataFrame(cancer_data['data'], columns = cancer_data['feature_names'])
# print(cancer_data['DESCR'])

#Split the data set into training data and test data
x = raw_data
y = cancer_data['target']
from sklearn.model_selection import train_test_split
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)

#Train the SVM model
from sklearn.svm import SVC
model = SVC()
model.fit(x_training_data, y_training_data)

#Make predictions with the model
predictions = model.predict(x_test_data)

#Measure the performance of our model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print(classification_report(y_test_data, predictions))
print(confusion_matrix(y_test_data, predictions))
