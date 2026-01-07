import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
from transformations import create_pipeline_stages, get_target_column

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OFFLINE_FILE = os.path.join(DATA_DIR, 'offline.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def train():    
    spark = SparkSession.builder \
        .appName("DiabetesClassificationOffline") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    try:
        df = spark.read.csv(OFFLINE_FILE, header=True, inferSchema=True)
        print(f"Loaded {df.count()} records")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    stages = create_pipeline_stages()
    target_col = get_target_column()
    
    evaluator = MulticlassClassificationEvaluator(
        labelCol=target_col, 
        predictionCol="prediction", 
        metricName="f1"
    )

    models_to_train = [
        (
            "LogisticRegression", 
            LogisticRegression(labelCol=target_col, featuresCol="scaledFeatures", maxIter=100),
            ParamGridBuilder()
                .addGrid(LogisticRegression.regParam, [0.01, 0.1, 0.5])
                .addGrid(LogisticRegression.elasticNetParam, [0.0, 0.5])
                .build()
        ),
        (
            "RandomForest", 
            RandomForestClassifier(labelCol=target_col, featuresCol="scaledFeatures"),
            ParamGridBuilder()
                .addGrid(RandomForestClassifier.numTrees, [10, 20, 30])
                .addGrid(RandomForestClassifier.maxDepth, [5, 10])
                .build()
        ),
        (
            "DecisionTree", 
            DecisionTreeClassifier(labelCol=target_col, featuresCol="scaledFeatures"),
            ParamGridBuilder()
                .addGrid(DecisionTreeClassifier.maxDepth, [5, 10, 15])
                .addGrid(DecisionTreeClassifier.minInstancesPerNode, [1, 5])
                .build()
        )
    ]

    best_model = None
    best_f1 = 0.0
    best_model_name = ""
    results = []

    for name, estimator, param_grid in models_to_train:
        pipeline = Pipeline(stages=stages + [estimator])
        
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3,
            parallelism=2
        )
        
        start_time = time.time()
        cv_model = cv.fit(df)
        training_time = time.time() - start_time
        
        predictions = cv_model.transform(df)
        f1 = evaluator.evaluate(predictions)
        
        
        results.append({
            'name': name,
            'f1': f1,
            'time': training_time
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = cv_model.bestModel
            best_model_name = name

    for result in results:
        print(f"{result['name']:25} | F1: {result['f1']:.4f} | Time: {result['time']:.2f}s")
    
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"BEST F1 SCORE: {best_f1:.4f}")
    print("=" * 80)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_path = os.path.join(MODEL_DIR, "best_diabetes_model")
    best_model.write().overwrite().save(model_path)

    spark.stop()

if __name__ == "__main__":
    train()
