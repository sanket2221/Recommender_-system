import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('ratings.csv')
movies =  pd.read_csv('movies.csv')
ratings = pd.merge(movies,ratings).drop(['genres','timestamp'],axis =1)

userRatings = ratings.pivot_table(index=['userId'],columns=['title'],values = 'rating')
userRatings = userRatings.dropna(thresh=10, axis=1).fillna(0,axis=1)

corrMatrix = userRatings.corr(method='pearson')

def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    return similar_ratings

movies = [('17 Again (2009)',3),('Amazing Spider-Man, The (2012)',4),('Iron Man (2008)',5),('Avengers: Age of Ultron (2015)',4),('Neighbors (2014)',4)]
similar_movies =pd.DataFrame()
for movie,rating in movies:
    similar_movies = similar_movies.append(get_similar(movie,rating),ignore_index=True)

similar_movies.sum().sort_values(ascending=False).head(20)


