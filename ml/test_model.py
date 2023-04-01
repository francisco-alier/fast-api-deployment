"""
Unit test of model.py module with pytest
author: Francisco Nogueira
"""

def test_train_model():
        """
    Check saved model is present
    """
    savepath = "./model/trained_model.pkl"
    if os.path.isfile(savepath):
        try:
            _ = pickle.load(open(savepath, 'rb'))
        except Exception as err:
            logging.error(
            "Testing saved model: Saved model does not appear to be valid")
            raise err
    else:
        pass
