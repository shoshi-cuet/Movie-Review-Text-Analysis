import csv
from argparse import HelpFormatter

from mrjob.job import MRJob
from mrjob.step import MRStep


class preProcess(MRJob):

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_divide),
            MRStep(mapper=self.mapper,)
        ]

    def mapper_divide(self, _, line):
        reader = csv.reader([line])
        for row in reader:
            if len(row) > 1:
                if row[0] != 'review':
                    if row[1] != 'help':
                        yield row[0], int(row[1])
                else:
                    yield 'review', 'help'

    def mapper(self, value, line):
        if(value != 0):
            yield value, line


if __name__ == '__main__':
    preProcess.run()
