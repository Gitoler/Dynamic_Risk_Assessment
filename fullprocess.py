"""
This fullprocess.py script is to Process all steps of automation of the
ML model scoring, monitoring, and re-deployment process.
author: Ung Van Tuan
date: July 10th 2023
"""
import json
import os
import re
import logging
import ingestion
import pandas as pd
import training
import subprocess
import deployment
import diagnostics
import reporting
from sklearn import metrics

logging.basicConfig(filename='logs/fullprocess.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Load the path variables from config.json
def get_config():
    with open('config.json','r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    prod_path = os.path.join(config['prod_deployment_path'])
    input_data_path = os.path.join(config['input_folder_path'])
    output_model_path = os.path.join(config['output_model_path'])
    return dataset_csv_path, test_data_path, prod_path, input_data_path, output_model_path

# proceeding the data
def check_model_drift():
    logging.info("Checking for model drift")
    # Check new data, which one should proceed.
    dataset_csv_path, _, prod_path, input_data_path, _ = get_config()
    with open(os.path.join(prod_path, 'latestscore.txt'), 'r') as file:
        f1_score = file.readline()
    deployed_f1_score = round(float(re.findall(r'\d\d*\.?\d+',f1_score)[0]), 2)
    data = pd.read_csv(os.path.join(dataset_csv_path,'finaldata.csv'))
    y_data = data.pop('exited')
    X_data = data.drop(['corporation'], axis=1)
    # Evaluate the model
    y_pred = diagnostics.model_predictions(data)
    new_f1_score = metrics.f1_score(y_data, y_pred)
    # Checking for model drift
    if round(new_f1_score, 2) >= deployed_f1_score:
        logging.info(f"There is no model drift New score: {new_f1_score}, \
                     old score: {deployed_f1_score}")
        return None
    else:
        # Check new data, which one should proceed again.
        logging.info("Retrained the model")
        training.train_model()
        # Re-deployment
        logging.info("Deploying the new model")
        deployment.store_model_into_pickle()
        # Diagnostics and reporting
        logging.info("Running the system diagnostics and reporting")
        reporting.score_model()
        subprocess.run(['python', 'apicalls.py'])

# Check and read new data
def check_new_data():
    logging.info("Checking for new data")
    print("something")
    # Read ingestedfiles.txt
    _, _, prod_path, input_data_path, _ = get_config()
    with open(os.path.join(prod_path,'ingestedfiles.txt'), 'r') as file:
        ingested_files = {line.strip('\n') for line in file.readlines()}
    ingested_files = set([file.split(" ")[2].split("/")[1] for file in ingested_files])
    # Get the data in sourcedata folder
    source_files = set([file for file in os.listdir(input_data_path) if file.endswith('.csv')])
    # Determine whether the source data folder
    if len(source_files.difference(ingested_files)) == 0:
        logging.info("There is no new data found")
        return None
    else:
        logging.info("Ingesting new data")
        ingestion.merge_multiple_dataframe()
        check_model_drift()

if __name__ == '__main__':
    check_new_data()