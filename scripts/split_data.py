import pandas as pd
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_FILE = os.path.join(DATA_DIR, 'diabetes_binary_health_indicators_BRFSS2015.csv')
OFFLINE_FILE = os.path.join(DATA_DIR, 'offline.csv')
ONLINE_FILE = os.path.join(DATA_DIR, 'online.csv')

def split_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        return

    train, test = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['Diabetes_binary']
    )

    train.to_csv(OFFLINE_FILE, index=False)
    test.to_csv(ONLINE_FILE, index=False)
    
if __name__ == "__main__":
    split_data()
