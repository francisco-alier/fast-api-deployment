# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import pandas as pd
import numpy as np
import pickle
import uvicorn
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

#Creating the input expectations and their types. We use all features
class Input(BaseModel):
    age: int
    capital_gain: int
    capital_loss: int
    education: str
    education_num: int
    fnlgt: int
    hours_per_week: int
    marital_status: str
    native_country: str
    occupation: str
    relationship: str
    race: str
    sex: str
    workclass: str 

    class Config:
        schema_extra = {
            "example": {
                "age": 30,
                "workclass": 'State-gov',
                "fnlgt": 66514,
                "education": 'Bachelors',
                "education_num": 10,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "capital_gain": 5000,
                "capital_loss": 0,
                "hours_per_week": 30,
                "native_country": 'Portugal'
            }
        }


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def greetings():
    return "Welcome to this amazing app! You will predict if a person earns more than 50k a year based on their characteristics."

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/predictions")
async def predict_sample(item: Input):
    data = {    'age': item.age,
                'capital-gain': item.capital_gain,
                'capital-loss': item.capital_loss,
                'education': item.education,
                'education-num': item.education_num,
                'fnlgt': item.fnlgt,
                'hours-per-week': item.hours_per_week,
                'marital-status': item.marital_status,
                'native-country': item.native_country,
                'occupation': item.occupation,
                'relationship': item.relationship,
                'race': item.race,
                'sex': item.sex,
                'workclass': item.workclass, 
                }
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
    
    return data


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)