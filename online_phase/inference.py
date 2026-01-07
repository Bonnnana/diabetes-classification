import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_json, struct
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_diabetes_model')

CHECKPOINT_DIR = "/tmp/spark_checkpoint_diabetes"

KAFKA_TOPIC_INPUT = 'health_data'
KAFKA_TOPIC_OUTPUT = 'health_data_predicted'
BOOTSTRAP_SERVERS = 'localhost:9092'


def get_schema():
    features = [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age',
        'Education', 'Income'
    ]
    return StructType([
        StructField(field, DoubleType(), True) for field in features
    ])


def inference():
    spark = (
        SparkSession.builder
        .appName("DiabetesClassificationOnline")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.0"
        )
        .config("spark.driver.memory", "4g")
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    loaded_model = PipelineModel.load(MODEL_PATH)

    df_raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC_INPUT)
        .option("startingOffsets", "latest")
        .load()
    )

    json_schema = get_schema()
    df_parsed = (
        df_raw
        .select(from_json(col("value").cast("string"), json_schema).alias("data"))
        .select("data.*")
    )

    predictions = loaded_model.transform(df_parsed)

    output_df = predictions.select(
        col("HighBP"),
        col("HighChol"),
        col("CholCheck"),
        col("BMI"),
        col("Smoker"),
        col("Stroke"),
        col("HeartDiseaseorAttack"),
        col("PhysActivity"),
        col("Fruits"),
        col("Veggies"),
        col("HvyAlcoholConsump"),
        col("AnyHealthcare"),
        col("NoDocbcCost"),
        col("GenHlth"),
        col("MentHlth"),
        col("PhysHlth"),
        col("DiffWalk"),
        col("Sex"),
        col("Age"),
        col("Education"),
        col("Income"),
        col("prediction").alias("predicted_diabetes_class")
    )

    kafka_output = output_df.select(
        to_json(struct(*[col(c) for c in output_df.columns])).alias("value")
    )

    query = (
        kafka_output.writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
        .option("topic", KAFKA_TOPIC_OUTPUT)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .outputMode("append")
        .start()
    )

    console_query = (
        output_df.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("numRows", 5)
        .start()
    )

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        query.stop()
        console_query.stop()
        spark.stop()


if __name__ == "__main__":
    inference()
