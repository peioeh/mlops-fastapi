'''
    Namespace to store and access the Dataframe object used in tests
    Author: Peio Lahirigoyen
    Date: April 2023
'''
import pandas as pd
import pytest

@pytest.fixture
def data():
    d = {'animal': ['cat', 'cat', 'cat', 'dog', 'dog', 'dog', 'dog'], 'age': [1, 5, 7, 2, 3, 10, 11], 'iscute': ['yes', 'no', 'no', 'yes', 'yes', 'no', 'no']}
    data = pd.DataFrame(data=d)
    return data

def pytest_configure():
    '''
        Creating a Dataframe object 'pytest.df' in Namespace
    '''
    pytest.x_train = None
    pytest.x_test = None
    pytest.y_train = None
    pytest.y_test = None
    pytest.rf = None
