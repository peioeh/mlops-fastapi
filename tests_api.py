from fastapi.testclient import TestClient
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {'msg': 'Welcome on the salary prediction API'}


def test_inference_1():
    response = client.post(
        "/inference/",
        headers={'Content-Type': 'application/json'},
        json={"age": "36",
              "workclass": "State-gov",
              "fnlgt": "212143",
              "education": "Bachelors",
              "education-num": "13",
              "marital-status": "Married-civ-spouse",
              "occupation": "Adm-clerical",
                            "relationship": "Wife",
                            "race": "White",
                            "sex": "female",
                            "capital-gain": "0",
                            "capital-loss": "0",
                            "hours-per-week": "20",
                            "native-country": "United-States"}
    )
    assert response.status_code == 200
    assert response.json() == {'Predicted salary': '>50K'}


def test_inference_2():
    response = client.post(
        "/inference/",
        headers={'Content-Type': 'application/json'},
        json={"age": "36",
              "workclass": "State-gov",
              "fnlgt": "212143",
              "education": "Bachelors",
              "education-num": "1",
              "marital-status": "Married-civ-spouse",
              "occupation": "Adm-clerical",
                            "relationship": "Wife",
                            "race": "White",
                            "sex": "female",
                            "capital-gain": "0",
                            "capital-loss": "0",
                            "hours-per-week": "20",
                            "native-country": "United-States"}
    )
    assert response.status_code == 200
    assert response.json() == {'Predicted salary': '<=50K'}
