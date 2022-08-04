from pyspark.ml import Pipeline
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Import the required packages
from pyspark.ml.feature import (CountVectorizer, RegexTokenizer, StringIndexer,
                                Tokenizer, VectorAssembler)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('text_binary_classifier').getOrCreate()

# Load data and rename column
df = spark.read.csv(
    "hdfs://namenode:9000/data/classification_processed_data.csv",
    header=True,
    inferSchema=True,
)

df.show()


stages = []
# 1. clean data and tokenize sentences using RegexTokenizer
regexTokenizer = RegexTokenizer(inputCol="review", outputCol="tokens", pattern="\\W+")
stages += [regexTokenizer]

# 2. CountVectorize the data
cv = CountVectorizer(inputCol="tokens", outputCol="token_features", minDF=2.0)#, vocabSize=3, minDF=2.0
stages += [cv]

# 3. Convert the labels to numerical values using binariser
indexer = StringIndexer(inputCol="help", outputCol="label")
stages += [indexer]

# 4. Vectorise features using vectorassembler
vecAssembler = VectorAssembler(inputCols=['token_features'], outputCol="features")
stages += [vecAssembler]

pipeline = Pipeline(stages=stages)
data = pipeline.fit(df).transform(df)

train, test = data.randomSplit([0.7, 0.3], seed = 2021)

# Initialise the model
nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

# Fit the model
model = nb.fit(train)

# Make predictions on test data
predictions = model.transform(test)
predictions.select("label", "prediction", "probability").show()

#Evaluation
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
accuracy = evaluator.evaluate(predictions)
print ("Model Accuracy: ", accuracy)

# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]).build()
cvEvaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")

# Run Cross-validation
cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid, evaluator=cvEvaluator)
cvModel = cv.fit(train)

# Make predictions on testData. cvModel uses the bestModel.
cvPredictions = cvModel.transform(test)

# Evaluate bestModel found from Cross Validation
evaluator.evaluate(cvPredictions)



