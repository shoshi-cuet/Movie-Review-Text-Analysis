import csv
import itertools
from itertools import combinations

import pandas as pd

data_original = pd.read_csv(r"hdfs://namenode:9000/data/recommendation_result.csv", header=None)
data_original.columns = ['movie', 'review_detail']

combinations_of_movies = list(set(combinations(data_original['review_detail'], 2)))
print(data_original.shape)

print(len(combinations_of_movies))
print(len(combinations_of_movies[0]))

data = pd.DataFrame(columns = ['review_id', 'movie_1', 'movie_2', 'review_detail','content'])

review_list_1 = []
review_list_2 = []
for movie in combinations_of_movies:
    movie_1 = movie[0]
    movie_2 = movie[1]
    review_list_1.append(movie_1)
    review_list_2.append(movie_2)

data['review_detail'] = review_list_1
data['content'] = review_list_2  
data['review_id'] = data.index 

data['movie_1'] = data.merge(data_original, on=['review_detail'], how='left')['movie']
genre_title = data_original.rename(columns={'review_detail': 'content'})
data['movie_2'] = data.merge(genre_title, on=['content'], how='left')['movie']


data.to_csv("hdfs://namenode:9000/data/lsh.csv")
