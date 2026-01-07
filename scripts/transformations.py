from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import col

def get_feature_columns():
    return [
        'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
        'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 
        'Education', 'Income'
    ]

def get_target_column():
    return 'Diabetes_binary'

def create_pipeline_stages():
    feature_cols = get_feature_columns()
    
    assembler = VectorAssembler(
        inputCols=feature_cols, 
        outputCol="features"
    )
    
    scaler = StandardScaler(
        inputCol="features", 
        outputCol="scaledFeatures", 
        withStd=True, 
        withMean=False
    )
    
    return [assembler, scaler]
