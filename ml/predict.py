"""
This file was created to help me predict an individual sample offline
"""

import pandas as pd
import numpy as np
import pickle

from ml.data import process_data
from ml.model import inference
#Cat features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

data = {"age": 60,
                "workclass": 'State-gov',
                "fnlgt": 2000,
                "education": 'Bachelors',
                "education-num": 10,
                "marital-status": "Never-married",
                "occupation": "Private",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "capital-gain": 10000000,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": 'United-States'}

row = pd.DataFrame(data, index=[0])

    #Load the artifacts
clf = pickle.load(open("./model/classifier.pkl", 'rb'))
encoder = pickle.load(open("./model/encoder.pkl", 'rb'))
lb = pickle.load(open("./model/lb.pkl", 'rb'))

    #processing the data
X_row, y_row, encoder_row, lb_row = process_data(
                                                      row, 
                                                      categorical_features=cat_features, 
                                                      training=False, 
                                                      encoder = encoder,
                                                      lb = lb
                                                            )
    
#predict
prediction = inference(clf, X_row)

#Return prediction
data["prediction"] = ">50K" if prediction[0] > 0.5 else "<=50K"

print(data["prediction"],prediction)
