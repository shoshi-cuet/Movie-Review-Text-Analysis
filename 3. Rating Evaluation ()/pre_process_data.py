import csv
from audioop import avg

import numpy as np
from mrjob.job import MRJob


class preProcess(MRJob):
    ratings = []

    def mapper(self, _, line):
        reader = csv.reader([line])
        for row in reader:
            if len(row):
                if row[0] != 'review_id':
                    if (len(row) > 7):
                        if len(row[0]) and len(row[2]) and len(row[3]):
                            yield row[2], (int(row[3]))


    def reducer(self, key, value):
        for v in value:
            self.ratings.append(v)
        yield key, round(np.mean(self.ratings))



if __name__ == '__main__':
    preProcess.run()
