import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn
%matplotlib inline

from sklearn.datasets import load_breast_cancer
raw_data = load_breast_cancer()

raw_data_frame = pd.DataFrame(raw_data['data'], columns = raw_data['feature_names'])
raw_data_frame.columns

#Standardize the data
from sklearn.preprocessing import StandardScaler
data_scaler = StandardScaler()
data_scaler.fit(raw_data_frame)
scaled_data_frame = data_scaler.transform(raw_data_frame)

#Perform the principal component analysis transformation
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(scaled_data_frame)

x_pca = pca.transform(scaled_data_frame)

print(x_pca.shape)
print(scaled_data_frame.shape)

#Visualize the principal components
plt.scatter(x_pca[:,0],x_pca[:,1])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

#Visualize the principal components with a color scheme
plt.scatter(x_pca[:,0],x_pca[:,1], c=raw_data['target'])
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

#Investigating at the principal components
pca.components_[0]
