import csv
import math

import numpy as np
from mrjob.job import MRJob
from mrjob.step import MRStep


class preProcess(MRJob):
    scores = []
    all_rmse = []
    def steps(self):
        return [
            MRStep(mapper=self.mapper_pre,
            reducer=self.reducer_pre),
            MRStep(mapper=self.mapper,
            reducer=self.reducer),
            MRStep(reducer=self.reducer_final)
        ]

    def mapper_pre(self, _, line):
        reader = csv.reader([line])
        for row in reader:
            if len(row):
                if (len(row) > 1):
                    if len(row[0]) and len(row[1]):
                        yield row[0], row[1]


    def reducer_pre(self, key, value):
        val = ','.join(value)
        yield key, val

    def mapper(self, value, line):
        ratings = line.split(',')
        mean_square_error = np.square(np.subtract(int(ratings[1]),int(ratings[0]))/int(ratings[1]))

        yield value, (mean_square_error)

    def reducer(self, key, values):
        for score in values:
            self.scores.append(score)
        mse = np.mean(self.scores)
        rmse = math.sqrt(mse)
        yield "RMSE ", (rmse)

    def reducer_final(self, key, values):
        for score in values:
            self.all_rmse.append(score)
        
        avg_rmse = round(np.mean(self.all_rmse),2)
        yield "RMSE : ", 100 - avg_rmse * 100

if __name__ == '__main__':
    preProcess.run()
