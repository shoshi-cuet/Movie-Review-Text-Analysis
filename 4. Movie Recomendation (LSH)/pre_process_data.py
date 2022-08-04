import csv
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
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
from mrjob.step import MRStep
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix

nltk.download('stopwords')

stop = nltk.corpus.stopwords.words('english')


class preProcess(MRJob):
    ratings = []

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_pre,
                reducer=self.reducer_pre),
            MRStep(mapper=self.mapper,)
                #    reducer=self.reducer )
            ]

    def mapper_pre(self, _, line):
        reader = csv.reader([line]) 
        for row in reader:
            if len(row):
                if row[0] != 'review_id':
                    if (len(row) > 7):
                        if len(row[0]) and len(row[2]) and len(row[7]):
                            row[7] = row[7].replace(',', ' ')
                            yield row[2], row[7]


    def reducer_pre(self, key, value):
        val = ""
        val = len(val) < 30000 and '\t'.join(value)
        yield key, val

    def mapper(self, key, line):
        yield key, line

if __name__ == '__main__':
    preProcess.run()
