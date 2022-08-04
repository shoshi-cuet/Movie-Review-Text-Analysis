# As Our First Task, Here we are generating Movie rating by analysing sentiment of each review.

# movie             rating
#   A                  7
#   B                  6
#   .                  .
#   .                  .


# For Running : `cd 'DAT 500 Final Project/1. Rating Generator (NLP)'/`

# `hadoop fs -rm -r /data/out`

# Then For Chunk Data :
`python3 rating_generator.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/test_data.csv --output-dir hdfs:///data/out`

# For Full Data :
`python3 rating_generator.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/combined_data.csv --output-dir hdfs:///data/out`

# For show the result in terminal:
`hadoop fs -text /data/out/part* | less`

