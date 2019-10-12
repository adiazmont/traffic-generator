"""
    Code taken from:
    https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Access the data from file in pandas format
df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'titmestamp'])

# Formatting data for the purposes of this example
movie_titles = pd.read_csv('Movie_Id_Titles')
df = pd.merge(df, movie_titles, on='item_id')

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['number_of_ratings'] = df.groupby('title')['rating'].count()

# ratings['rating'].hist(bins=60)
# sns.jointplot(x='rating', y='number_of_ratings', data=ratings)
# plt.show()

# After viewing the data and visible patterns/correlations
# decided to use 'ratings' as the relevant value from each
# item to be used by the recommender system
movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')

# As an example, two movies are taken to find which movies
# to recommend
AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']

# Find the correlation within the selected movies and the dataset
# This corrwith function is the mathematical part of the recommender
# system. Hence, this is the part that gets improved with complex
# tools (i.e., ML, DL).
similar_to_air_force_one = movie_matrix.corrwith(AFO_user_rating)
similar_to_contact = movie_matrix.corrwith(contact_user_rating)

# Remove NaN values from the correlations
corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_contact.dropna(inplace=True)
corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace=True)

# Join the structures of the correlations with the number
# of ratings for each movie
corr_AFO = corr_AFO.join(ratings['number_of_ratings'])
corr_contact = corr_contact.join(ratings['number_of_ratings'])

# Show/recommend movies with high correlation values for
# which the number of ratings is higher than 100
print(corr_AFO[corr_AFO['number_of_ratings'] > 100].sort_values(by='correlation', ascending=False).head(10))
print(corr_contact[corr_contact['number_of_ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10))
