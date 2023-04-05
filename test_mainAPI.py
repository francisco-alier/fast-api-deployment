"""
Unit test of main.py API module with pytest
author: Francisco Nogueira
"""

from fastapi.testclient import TestClient
import json
from main import app

client = TestClient(app)


def test_root():
    """
    Test welcome message - GET
    """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to this amazing app! You will predict if a person earns more than 50k a year based on their characteristics."


def test_inference_case1():
    """
    Test model inference output for one case - POST
    """
    sample =  {"age": 30,
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
                "native_country": 'Portugal'}

    data = json.dumps(sample)

    r = client.post("/predictions", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["workclass"] == 'State-gov'

    # test prediction vs expected label
    prediction = r.json()["prediction"]
    assert prediction == '<=50K'


def test_inference_case2():
    """
    Test model inference output for one case - POST
    """
    sample =  {"age": 60,
                "workclass": 'State-gov',
                "fnlgt": 2000,
                "education": 'Bachelors',
                "education_num": 10,
                "marital_status": "Never-married",
                "occupation": "Private",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital_gain": 900000,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": 'United-States'}

    data = json.dumps(sample)

    r = client.post("/predictions/", data=data )

    # test response and output
    assert r.status_code == 200
    assert r.json()["native-country"] == 'United-States'

    # test prediction vs expected label
    prediction = r.json()["prediction"]
    assert prediction == '<=50K'