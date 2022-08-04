import csv

import pandas as pd
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol
from mrjob.step import MRStep
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Classification(MRJob):
    # OUTPUT_PROTOCOL = RawValueProtocol

    def mapper_init(self):
        url = r"C:\Users\Rimel\Desktop\DAT 500 Final Project\Dataset\Test Data\classification_processed_data.csv"

        data = pd.read_csv(url, header=None)


        training_data, testing_data = train_test_split(
            data, test_size=0.30)

        training_data = training_data.iloc[:,-2:]
        testing_data = testing_data.iloc[:,-2:]

        self.train_x = training_data.iloc[:, 1:]
        self.train_y = training_data.iloc[:, :-1]

        self.test_x = testing_data.iloc[:, 1:]
        self.test_y = testing_data.iloc[:, :-1]

    def mapper(self, _, line):
        yield "1", "2"
        reader = csv.reader([line])
        for row in reader:
            if len(row):
                if row[0] == 'review_id':
                    count_vect = CountVectorizer()
                    X_train_counts = count_vect.fit_transform(self.train_x)

                    # # transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
                    tfidf_transformer = TfidfTransformer()
                    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

                    knn = KNeighborsClassifier(n_neighbors=7)

                    # # training our classifier ; train_data.target will be having numbers assigned for each category in train data
                    clf = knn.fit(X_train_tfidf, self.train_y)
                    

                    predicted = clf.predict(self.test_x)

                    yield "results", predicted


if __name__ == '__main__':
    Classification.run()
