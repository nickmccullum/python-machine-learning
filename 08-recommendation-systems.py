#Data imports
import pandas as pd
import numpy as np

#Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Import the data
raw_data = pd.read_csv('u.data', sep = '\t', names = ['user_id', 'item_id', 'rating', 'timestamp'])
movie_titles_data = pd.read_csv('Movie_Id_Titles')

#Merge our two data sources
merged_data = pd.merge(raw_data, movie_titles_data, on='item_id')
merged_data.columns

#Calculate aggregate data
merged_data.groupby('title')['rating'].mean().sort_values(ascending = False)
merged_data.groupby('title')['rating'].count().sort_values(ascending = False)

#Create a DataFrame and add the number of ratings to is using a count method
ratings_data = pd.DataFrame(merged_data.groupby('title')['rating'].mean())
ratings_data['# of ratings'] = merged_data.groupby('title')['rating'].count()

#Make some visualizations
sns.distplot(ratings_data['# of ratings'])
sns.distplot(ratings_data['rating'])

#Create the ratings matrix and get user ratings for `Return of the Jedi (1983)` and `Toy Story (1995)`
ratings_matrix = merged_data.pivot_table(index='user_id',columns='title',values='rating')
star_wars_user_ratings = ratings_matrix['Return of the Jedi (1983)']
toy_story_user_ratings = ratings_matrix['Toy Story (1995)']
ratings_matrix.corrwith(toy_story_user_ratings)['Return of the Jedi (1983)']

#Calculate correlations and source recommendations
correlation_with_star_wars = pd.DataFrame(ratings_matrix.corrwith(star_wars_user_ratings))
correlation_with_star_wars.dropna().sort_values(0, ascending = False).head(15)

#Add the number of ratings and rename columns
correlation_with_star_wars = correlation_with_star_wars.join(ratings_data['# of ratings'])
correlation_with_star_wars.columns = ['Corr. With SW Ratings', '# of Ratings']
correlation_with_star_wars.index.names = ['Movie Title']

#Get new recommendations from movies that have more than 50 ratings
correlation_with_star_wars[correlation_with_star_wars['# of Ratings'] > 50].sort_values('Corr. With SW Ratings', ascending = False).head(10)
