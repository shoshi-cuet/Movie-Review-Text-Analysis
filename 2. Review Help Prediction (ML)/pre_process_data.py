import csv

import nltk
from mrjob.job import MRJob
from mrjob.step import MRStep

nltk.download('words')

words = set(nltk.corpus.words.words())


class preProcess(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_divide),
            MRStep(mapper=self.mapper)
        ]

    def mapper_divide(self, _, line):
        reader = csv.reader([line])
        for row in reader:
            if len(row):
                if row[0] != 'review_id':
                    if (len(row) > 8):
                        if len(row[0]) and len(row[7]) and len(row[8]):
                            row[7] = " ".join(w for w in nltk.wordpunct_tokenize(row[7]) if w.lower() in words or not w.isalpha())
                            row[7] = row[7].replace('"', '').replace('\t', '').replace("'", '')
                            row[7] = row[7].strip()
                            row[8] = row[8].strip()
                            yield row[7], row[8]
                else:
                    yield 'review', 'help'

    def mapper(self, value, line):
        helpful = line.split(',')
        help = 'help'
        if (len(helpful) > 1):
            helpful[0] = helpful[0].replace('[', '')
            helpful[0] = helpful[0].replace("'", '')
            helpful[1] = helpful[1].replace(']', '')
            helpful[1] = helpful[1].replace("'", '')
            if(round(int(helpful[1])/2) > int(helpful[0])):
                help = 0
            else:
                help = 1

        yield value, help


if __name__ == '__main__':
    preProcess.run()
