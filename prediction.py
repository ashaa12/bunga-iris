import joblib
import numpy as np

model = joblib.load("random_forest_model.joblib")

def predict(data):
    result = model.predict(data)
    return result
