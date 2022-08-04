# For Running : `cd 'DAT 500 Final Project/3. Rating Evaluation ()'/`

# `hadoop fs -rm -r /data/out`

# Pre processing :
`python3 pre_process_data.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/combined_data.csv --output-dir hdfs:///data/out`

# Then convert part* file into 'evaluation_processed_data.csv' in 'hadoop fs /data' folder.

# For show the result in terminal:
`hadoop fs -text /data/evaluation_processed_data.csv | less`

# Then for Map reduce file Evaluation :
`python3 rmse_evaluation.py --hadoop-streaming-jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar -r hadoop hdfs:///data/evaluation_processed_data.csv hdfs:///data/generated_rating_data.csv`
