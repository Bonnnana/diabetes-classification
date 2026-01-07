import json
import time
import pandas as pd
import os

from kafka import KafkaProducer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SOURCE_FILE = os.path.join(DATA_DIR, 'offline.csv')

KAFKA_TOPIC = 'health_data'

producer = KafkaProducer(bootstrap_servers='localhost:9092', security_protocol="PLAINTEXT")

df = pd.read_csv(SOURCE_FILE)

if 'Diabetes_binary' in df.columns:
    df = df.drop(columns=['Diabetes_binary'])

for index, row in df.iterrows():
    record = row.to_dict()
    
    print(json.dumps(record))
    producer.send(
        topic="health_data",
        value=json.dumps(record).encode("utf-8")
    )
    time.sleep(0.05)
