"""
This app.py script is to set up a Flask API
author: Ung Van tuan
date: July 10th 2023
"""
from flask import Flask, request
import pandas as pd
import subprocess
import logging
import diagnostics as diag
import json
import os

# Initialize logging
logging.basicConfig(filename='logs/api.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Load the path variables from config.json
app = Flask(__name__)
app.secret_key = '192b9bdd22-ab9ed4d12e236c78-9a393ec15-f71bbf5dc987d547-27823'
def get_config():
    with open('config.json','r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    prod_deployment_path = config["prod_deployment_path"]

    return prod_deployment_path, dataset_csv_path

@app.route("/")
def index():
    logging.info("Call the index endpoint")
    return "The Dynamic Risk Assessment System API!"

# Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    """
    Endpoint that allow a client to get the prediction of an iput file
    Output:
     y_pred: List
    """
    logging.info("The predict endpoint have been call")
    dataset_file_path = request.args.get('dataset_file_path')
    dataset = pd.read_csv(dataset_file_path)
    y_test = dataset.pop('exited')
    # Call the prediction function
    y_pred = diag.model_predictions(dataset)
    print(y_pred)
    # Add return value for prediction outputs
    return y_pred.tolist()

# Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():
    """
    Endpoint that runs the script scoring.py and
    gets the score of the deployed model
    Output:
    f1_score: float
    """
    logging.info("The score endpoint have been call")
    # Check the score of the deployed model
    f1_score = subprocess.run(['python', 'scoring.py'], capture_output=True).stdout
    f1_score = f1_score.decode().strip()
    # Add return value (a single F1 score number)
    return f1_score

# Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    """
    Endpoint that return the summary statistics
    Output:
    json_stats: json
        The summary statistics in json format
    """
    logging.info("The stats endpoint have been call")
    # Check means, medians, and modes
    json_stats = json.dumps(diag.dataframe_summary(), indent = 4)
    # Return a list of all calculated summary statistics
    return json_stats

# Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    """
    Endpoint that return the timing, missing data, and dependency
    Output:
        dict: missing percentage, execution time and outdated packages
    """
    logging.info("The diagnostics endpoint have been call")
    # check timing, the outdated library and percent NA values
    diag_result = {
        'missing_data_percentage': diag.missing_data(),
        'execution_time': diag.execution_time(),
        'outdated_packages': diag.outdated_packages_list()
    }
    # Convert the dictionary into json
    json_diag = json.dumps(diag_result, indent=4)
    # Add return value for all diagnostics
    return json_diag

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)