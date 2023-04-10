import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference
import pandas as pd
import joblib

app = FastAPI()

rf = joblib.load('model/random_forest.joblib')
encoder = joblib.load('model/encoder.joblib')
lb = joblib.load('model/lb.joblib')

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


def to_hyphen(string: str) -> str:
    return '-'.join(word for word in string.split('_'))


class IndividualAttributes(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        alias_generator = to_hyphen
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "age": 36,
                "workclass": "State-gov",
                "fnlgt": 212143,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Adm-clerical",
                "relationship": "Wife",
                "race": "White",
                "sex": "female",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 20,
                "native-country": "United-States"
            }
        }


# Use POST action to send data to the server
@app.post("/inference/")
async def api_inference(individual: IndividualAttributes):
    x_dict = {k: [v] for k, v in individual.dict(by_alias=True).items()}
    X = pd.DataFrame.from_dict(x_dict)
    X_infer, _, _, _ = process_data(
        X, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    # Model inference
    y_pred = inference(rf, X_infer)
    res = "<=50K" if y_pred[0] <= 0.5 else ">50K"
    return {"Predicted salary": res}


# GET on the root giving a welcome message.
@app.get("/")
async def welcome():
    return {'msg': 'Welcome on the salary prediction API'}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
