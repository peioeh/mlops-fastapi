# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.metrics import slice_metrics
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
file_handler = logging.FileHandler('slice_output.txt')
logger.addHandler(file_handler)


# Add code to load in the data.
data = pd.read_csv('../data/census.csv', sep=',')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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

# Proces the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train the model.
rf = train_model(X_train, y_train)

# Save model
joblib.dump(rf, '../model/random_forest.joblib')
joblib.dump(encoder, '../model/encoder.joblib')
joblib.dump(lb, '../model/lb.joblib')

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Model inference
y_pred = inference(rf, X_test)

# Determine the classification metrics.
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logger.info(f'Classification metrics: precision={precision}, recall={recall}, fbeta={fbeta}\n')

# Computes model metrics on slices of the data (when the value of marital-status is held fixed)
logger.info('Compute metrics when the value of marital-status is held fixed:')
metrics = slice_metrics(rf, test, cat_features, encoder, lb, 'marital-status')
for feature_val in metrics:
    precision, recall, fbeta = metrics[feature_val]
    logger.info(f'{feature_val}: precision={precision}, recall={recall}, fbeta={fbeta}')






