import os
os.chdir("/Applications/VSCode/MLProjs/movie_recs")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index): #finds name of movie through index
	return movie_data[movie_data.index == index]["title"].values[0]

def get_index_from_title(title): #finds index of user's favorite movie
	return movie_data[movie_data.title == title]["index"].values[0]

movie_data = pd.read_csv("movie_dataset.csv") #reads csv file
features = ['keywords','cast','genres','director'] #picking features for comparison
for feature in features:
	movie_data[feature] = movie_data[feature].fillna('') #fills Nan values with empty string

def combine_features(row): #combines selected features
	return row['keywords'] + ' ' + row['cast'] + ' ' + row['genres'] + ' ' + row['director']

movie_data["combined_features"] = movie_data.apply(combine_features,axis=1) #applies combine features function to all rows

# create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie_data["combined_features"])
cos_sim = cosine_similarity(count_matrix)

fav_movie = input("What movie do you like? ") 
movie_index = get_index_from_title(fav_movie)
similar_movies = list(enumerate(cos_sim[movie_index]))
sorted_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True) #sorts movies in descending order based on cosine similarity

i=0
for movie in sorted_movies: #prints first 50 movies
	print(get_title_from_index(movie[0]))
	i=i+1
	if i>50:
		break
