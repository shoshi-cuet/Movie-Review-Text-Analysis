import csv
from asyncore import read

import spacy

nlp = spacy.load("en_core_web_sm")

import numpy as np
from mr3px.csvprotocol import CsvProtocol
from mrjob.job import MRJob
from mrjob.step import MRStep
from textblob import TextBlob


class RatingGenerator(MRJob):
    #OUTPUT_PROTOCOL = CsvProtocol  # write output as CSVs

    def steps(self):
        return [
            MRStep(
                mapper=self.mapper_pre,
                reducer=self.reducer_pre),
            MRStep(mapper=self.mapper,
                   reducer=self.reducer )
            ]

    def mapper_pre(self, _, line):
        reader = csv.reader([line])
        for row in reader:
            if len(row):
                if row[0] != 'review_id':
                    if (len(row) > 7):
                        if len(row[0]) and len(row[2]) and len(row[7]):
                            yield row[2], row[7]

    def reducer_pre(self, key, value):
        val = '\t'.join(value)
        yield key, val

    def mapper(self, key, line):
        reviews = line.split('\t')
        score_per_review = []
        for review in reviews:
            sentence_scores = []
            review = nlp(review)
            for sentence in review.sents:
                tokens = nlp(sentence.text)
                scores = []
                for token in tokens:
                    if(token.pos_== 'NOUN' or token.pos_ == "ADV" or token.pos_ == "VERB" or token.pos_ == "ADJ"):
                        o_tag = ""
                        if(token.pos_ == 'NOUN'):
                            o_tag = 'n'
                        elif(token.pos_ == 'VERB'):
                            o_tag = 'v'
                        elif(token.pos_ == 'ADV'):
                            o_tag = 'r'
                        else:
                            o_tag = 's'

                        #ss_net = list(swn.senti_synsets(token.text , o_tag))

                        syn_o_word = TextBlob(token.text)

                        score = syn_o_word.sentiment.polarity * syn_o_word.sentiment.subjectivity
                        if score > 0:
                            scores.append(score)
                if len(scores): 
                    sentence_scores.append(sum(scores)/len(scores))
            
            if (len(sentence_scores)):
                score_per_review.append(sum(sentence_scores)/len(sentence_scores))
        
        if len(score_per_review):
            avg = sum(score_per_review) / len(score_per_review)
            rating = round((avg * 10) - 5 * (avg - 1))

            yield key, rating
        
    def reducer(self, key, value):
        yield key, sum(value)

if __name__ == '__main__':
    RatingGenerator.run()
