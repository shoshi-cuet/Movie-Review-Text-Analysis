import nltk
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
import collections
import itertools
import json
import random
import re
import string
from collections import Counter
from itertools import combinations

import nltk
import numpy as np
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix

stop = nltk.corpus.stopwords.words('english')

data = pd.read_csv(r"hdfs://namenode:9000/data/lsh.csv")

data['content'] = data['content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data['review_detail'] = data['review_detail'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

def createNgrams(articles):
  ngrams_list = []
  for i,row in articles.iterrows():
    
    content_tokens = word_tokenize(row['content'])
    content_tokens = [w.lower() for w in content_tokens]
    content_table = str.maketrans('', '', string.punctuation)
    stripped_tokens = [w.translate(content_table) for w in content_tokens]
    # remove remaining tokens that are not alphabetic
    content_words = [word for word in stripped_tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in content_words if not w in stop_words]


    title_tokens = word_tokenize(row['review_detail'])
    title_tokens = [w.lower() for w in title_tokens]
    title_table = str.maketrans('', '', string.punctuation)
    stripped_tokens = [w.translate(title_table) for w in title_tokens]
    # remove remaining tokens that are not alphabetic
    title_words = [word for word in stripped_tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    title_words = [w for w in title_words if not w in stop_words]


    total_content = list(nltk.ngrams(content_words, 2))
    total_title = list(nltk.ngrams(title_words, 2))

    total= list(set(total_content + total_title))
    ngrams_list.append(total)

  articles['ngrams'] = ngrams_list
  return articles


data1 = createNgrams(data)
data1


shingles =[]
for i in data1.ngrams:
    shingles.extend(i)
counter= collections.Counter(shingles).most_common(10000)

unique_shingles =[]
for i in counter:
    unique_shingles.append(i[0])

def getBinaryMatrixnp(movies, shingles):
    bin_matrix = np.zeros([len(shingles),len(movies)])

    for i, gram in enumerate(shingles):
        t = movies[movies['ngrams'].apply(lambda x: gram in x)]
        bin_matrix[i, list(t['review_id'])] = 1
    return bin_matrix

binary_matrix = getBinaryMatrixnp(data1, unique_shingles)

def getHashFunctionValues(numrows, numhashfunctions):
    hash_mat = np.zeros((numrows,numhashfunctions))
    rowindex=np.array([i for i in range(numrows)])
    for i in range(numrows):
        np.random.shuffle(rowindex)
        for j in range(numhashfunctions):
            hash_mat[i,j]=(hash(rowindex[j])%numrows)
    return hash_mat.T
    #return a matrix with hash values


hash_val_matrix=getHashFunctionValues(10000,100)


def getMinHashsig(binary_matrix, hash_val_matrix):
    signature_matrix=np.full((hash_val_matrix.shape[0],binary_matrix.shape[1]),np.inf)
    #binary_matrix = binary_matrix.to_numpy()
    rows, cols = np.where(binary_matrix == 1)
    
    for j in np.arange(0, rows.shape[0]):
        r,c = rows[j], cols[j]
        for i in np.arange(0, hash_val_matrix.shape[0]):
            if hash_val_matrix[i,r] < signature_matrix[i,c]:
                signature_matrix[i,c] = hash_val_matrix[i,r]
                
    return signature_matrix

signature_matrix = getMinHashsig(binary_matrix, hash_val_matrix)

def getLSH(signature_matrix, hashfunctions, num_bands):
    buckets = {}
    
    r = signature_matrix.shape[0] / num_bands
    bands = np.split(signature_matrix, r)
    
    for i in range(len(bands)):
        band = bands[i].T
        for j in np.arange(0, band.shape[0]):            
            hsh = hash("".join([str(v) for v in band[j]]))
            if hsh in buckets:
                buckets[hsh].append(j)
            else:
                buckets[hsh] = [j]           
    #return lsh buckets or hash table
    return buckets

def jaccard_similarity(cantidate_1, cantidate_2):
    intersection = len(list(set(cantidate_1).intersection(cantidate_2)))
    union = (len(cantidate_1) + len(cantidate_2)) - intersection
    return float(intersection / union)

buckets = getLSH(signature_matrix, hash_val_matrix, num_bands = 10)


def find_cantidate_pairs(buckets):
    cantidate_pairs = []
    
    count = 0
    for key, val in buckets.items():
        count = count + 1
        if len(val) > 1:
            val = list(set(val))
            
            new_pairs = list(itertools.combinations(val, 2))
            cantidate_pairs.extend(new_pairs)
            
    return list(set(cantidate_pairs))

cantidate_pairs = find_cantidate_pairs(buckets)

def plot_thr(candidate_pairs, b):

    r = signature_matrix.shape[0] / b
    sim = []
    prob = []
    for pair in cantidate_pairs[:10000]:
        cantidate_1 = signature_matrix[:, pair[0]]
        cantidate_2 = signature_matrix[:, pair[1]]

        s = jaccard_similarity(cantidate_1, cantidate_2)
        p = 1 - (1-(s**r))**b

        sim.append(s)
        prob.append(p)

    return sim, prob

sim, prob = plot_thr(cantidate_pairs, b = 10)
sim1 = np.sort(sim)[::-1]
prob1 = np.sort(prob)[::-1]


def choose_candidates(cantidate_pairs, signature_matrix, threshold):
    chosen_cantidates = [] 

    for pair in cantidate_pairs:
        cantidate_1 = signature_matrix[:, pair[0]]
        cantidate_2 = signature_matrix[:, pair[1]]
        
        if(jaccard_similarity(cantidate_1, cantidate_2) >= threshold):
            chosen_cantidates.append(pair)
            
    return chosen_cantidates  

chosen_cantidates_1 = choose_candidates(cantidate_pairs, signature_matrix, threshold = 0.14)


candidate_pairs_df =  pd.DataFrame(chosen_cantidates_1)


movie_list = pd.DataFrame(data.iloc[candidate_pairs_df[0]]['movie_1']).reset_index(drop=True)


similar_movie_list =pd.DataFrame(data.iloc[candidate_pairs_df[1]]['movie_2']).reset_index(drop=True)


submission = movie_list.join(similar_movie_list)
submission.groupby('movie_1').value_counts()

submission.to_csv('hdfs://namenode:9000/data/recommendation_result.csv', index=False, header=False)
