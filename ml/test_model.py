"""
Unit test of model.py module with pytest
author: Francisco Nogueira
"""

import logging
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

from ml.data import process_data
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

DATA_PTH = "./data/census.csv"
MODEL_PTH = "./model/classifier.pkl"
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

@pytest.fixture(name="data")
def load_data():
    """
    Fixture just to load data
    """
    df = pd.read_csv(DATA_PTH)
    return df


@pytest.fixture(name="model")
def load_model():
    """
    Fixture to load model
    """
    clf = pickle.load(open(MODEL_PTH, 'rb'))
    return clf


def test_import_data(data):
    """
    Test presence and shape of dataset file
    """
    # Check the df shape
    try:
        assert data.shape[0] > 0 and data.shape[1] > 0

    except AssertionError as err:
        logging.error(
        "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_model(model):
   
    """ Check model type """
    try:
        assert isinstance(model, GradientBoostingClassifier)

    except AssertionError as err:
        logging.error(
          "Testing model: It is not the correct model type"  
        )


def test_process_data(data):

    """ Test the data split """

    try:
        train, _ = train_test_split(data, test_size=0.20)
        X, y, _, _ = process_data(train, cat_features, label='salary')
        assert len(X) == len(y)

    except AssertionError as err:
        logging.error(
            "Testing tran test splt - the lenghts differ"
        )