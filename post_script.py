import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

API_ENDPOINT = 'https://infer-salary.onrender.com/inference/'


# data to be sent to api
data = {
                "age": "36",
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
                "native-country": "United-States"
            }

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, json=data, verify=False)

# extracting response text
logger.info(f'Status code: {r.status_code}')
logger.info(f'Content: {r.text}')