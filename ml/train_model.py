# Script to train machine learning model.


from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_metrics_on_slices
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import logging
import os

# Initialize logging
logging.basicConfig(filename='train_model.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Loading the data
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder = encoder, lb = lb
)

#In case model already exists
if os.path.isfile("../model/classifier.pkl"):
    clf = pickle.load(open("../model/classifier.pkl", 'rb'))

else:
    # Train and save a model.
    clf = train_model(X_train, y_train)
    pickle.dump(clf, open("../model/classifier.pkl", 'wb'))
    pickle.dump(encoder, open("../model/encoder.pkl", 'wb'))
    logging.info("Model & encoder were saved to disk in folder model")

#Inference functions
preds = inference(clf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")

# Compute performance on slices for categorical features
for feature in cat_features:
    performance_df = compute_metrics_on_slices(X_test, y_test, feature, clf)
    logging.info(f"Performance on slice {feature}")
    logging.info(performance_df)