from .data import process_data
from .model import compute_model_metrics, inference
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def slice_metrics(rf, test, cat_features, encoder, lb, feature):
    """
    :param rf: random forest model
    :param test: test dataset
    :param cat_features: : list of categorical features
    :param encoder: OneHotEncoder
    :param lb: learn.preprocessing._label.LabelBinarizer
    :param feature: the feature we fix the value
    :return: a dictionary with precision, recall, and F1 for each value of the categorical feature
    """
    metrics = {}
    for feature_val in test[feature].unique():
        # Proces the test data with the process_data function.
        X_test, y_test, _, _ = process_data(
            test[test[feature] == feature_val], categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )

        # Model inference
        y_pred = inference(rf, X_test)

        # Determine the classification metrics.
        metrics[feature_val] = compute_model_metrics(y_test, y_pred)
    return metrics
