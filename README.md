# Project3 - using FastAPI to deploy a Machine Learning Model

## Submission Link

* Github repo: https://github.com/francisco-alier/fast-api-deployment

## Overview
This is the repository for project #3 of the udacity nanodegree for Dev Ops Engineering. It involved several steps:

* Train a ML model on a classification task to predict sand if it is over 50k. It is a classification task
* Expose the model for inference using a FastAPI app
* Deploy the app using Render
* Implement Continuous Integration / Continuous Deployment workflow using Github actions, github repository and Heroku integration with Github. The app is only deployed if integrated, automated, tests are validated by Github actions upon modifications


## Model

The model card can be found [here](model_card_template.md)

If you wish to train the model you can simply run the code:
```bash
python src/train_model.py
```
If you wish to run the entire ML pipeline you can use the following code:
```bash
python main.py
```
Deployment
This app can also be tested live on Render. 
It is fully integrated in with CI and a specified Github action:


Currently there is no setup on this but one can create a web service and then use the request library to generate some query's to the app as it indicates in the following code that uses a POST request:
```bash
python api_request.py
```