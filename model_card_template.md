# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The provided task is a classification task that predicts wheter a person earns more than 50k a year. The created model is a Gradient Boosting Machine from sklearn with tunned hyperparameters which are the following:


* learning_rate: 0.1
* max_depth: 6
* min_samples_leaf: 1
* min_samples_split: 5
* n_estimators: 80
* random_state : 42
* subsample : 1.0

## Intended Use
The model can be used to predict these specific classification task. Currently it is only using a pre-selected attributes so eventually it can be modified to inclue more or less features.

## Training Data
The dataset was obtained from the uci public repository and its extraction was done by Barry Becker from the 1994 Census database. The dataset has 32561 rows and 15 columns, out of which 8 are categorical, 6 numerical and 1 the target (>50k or <=50k). More information can be found (here)[https://archive.ics.uci.edu/ml/datasets/census+income]

## Evaluation Data
We applied categorical encoding to the necessary features of that nature by using the same encoders as used on the training set. Those encoders were the label binarizer for the target and One Hot Encoder for the categorical features.
We used 20% of data for testing our model.

## Metrics
The metrics evaluated were the fbeta, precision and recall,
On the test set we achieve the following values:
* precision:0.797
* recall:0.670
* fbeta:0.728

## Ethical Considerations
Special considerations should be acounted for regarding some sensitive variables such as sex, race and workclass. Ensure that the model has a fair representation on the population.

## Caveats and Recommendations
The database is quite outdates (1994) so a refresh on the census data might be better representative of the current population
