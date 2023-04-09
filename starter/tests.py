import pytest
from sklearn.ensemble import RandomForestClassifier

import conftest
from starter.ml.data import process_data
from starter.ml.model import train_model, inference
from sklearn.model_selection import train_test_split
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def test_process_data(data):
	"""
	test processing of data
	"""
	train, test = train_test_split(data, test_size=0.05)
	cat_features = [
		"animal"
	]

	# Proces the train data with the process_data function.
	X_train, y_train, encoder, lb = process_data(
		train, categorical_features=cat_features, label="iscute", training=True
	)
	try:
		assert X_train.shape[1] == 3
	except AssertionError as err:
		logging.error(f'Could not check X_train shape')
		raise err
	pytest.x_train = X_train
	pytest.y_train = y_train

	# Proces the test data with the process_data function.
	X_test, y_test, _, _ = process_data(
		test, categorical_features=cat_features, label="iscute", training=False, encoder=encoder, lb=lb
	)
	try:
		assert X_test.shape[1] == 3
	except AssertionError as err:
		logging.error(f'Could not check X_test shape')
		raise err
	pytest.x_test = X_test
	pytest.y_test = y_test


def test_train_model():
	"""
	test training of the model
	"""
	try:
		rf = train_model(pytest.x_train, pytest.y_train)
		assert isinstance(rf, RandomForestClassifier)
	except AssertionError as err:
		logging.error('Could not check the type of the trained model')
		raise err
	pytest.rf = rf


def test_inference():
	"""
	test inference function
	"""
	try:
		# Any animal aged 1 is cute!
		y_pred = inference(pytest.rf, np.array([[1.,1.,0.], [1.,0.,1.]]))
		assert y_pred[0] == 1
		assert y_pred[1] == 1
	except AssertionError as err:
		logging.error('Could not check the inferred values')
		raise err