# As our second task we are using spark here for classification review.

# First we are processing the data to convert the column helpful -> [ 3,5 ] (where 3 people out of 5 marked the review as helpful) into new column help with 1 ( helpful) or 0 (not helpful)


# For Running : `cd 'DAT 500 Final Project/2. Review Help Prediction (ML)'/`

# `hadoop fs -rm -r /data/out`

# Pre processing :
`python3 pre_process_data.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/combined_data.csv --output-dir hdfs:///data/out`

# Then convert part* file into 'classification_processed_data.csv' in 'hadoop fs /data' folder.

# For show the result in terminal:
`hadoop fs -text /data/classification_processed_data.csv | less`

# then For Classification in MRJOb :
`python3 text_classification.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/generated_rating_data.csv`

# For Classification in Spark :
`python3 naive_bayes_spark.py`

