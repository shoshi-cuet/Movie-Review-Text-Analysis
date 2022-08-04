# For Running : `cd 'DAT 500 Final Project/4. Movie Recomendation (LSH)'/`

# `hadoop fs -rm -r /data/out`

# Pre processing :
`python3 pre_process_data.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/combined_data.csv --output-dir hdfs:///data/out`

# Then convert part* file into 'recommendation_processed_data.csv' in 'hadoop fs /data' folder.

# For show the result in terminal:
`hadoop fs -text /data/recommendation_processed_data.csv | less`

# Then for LSH :
`python3 lsh.py`

# Final result will be in hadoop fs /data/recommendation_result.csv file.


