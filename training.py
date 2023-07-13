"""
This training.py script is to ingest data.
author: Ung Van Tuan
date: July 10th 2023
"""

import pandas as pd
import pickle
import logging
import os
from sklearn.linear_model import LogisticRegression
import json

# Initialize logging
logging.basicConfig(filename='logs/training.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Load the path variables from config.json
def get_config():
    with open('config.json','r') as f:
        config = json.load(f)

    dataset_csv_path = os.path.join(config['output_folder_path'])
    model_path = os.path.join(config['output_model_path'])

    return dataset_csv_path, model_path


# Create function to train the model
def train_model():
    """
    Function to complete model training
    """
    # Get the needed path
    dataset_csv_path, model_path = get_config()
    # Read the dataset
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    y_data = data.pop('exited')
    X_data = data.drop(['corporation'], axis=1)
    # Use this logistic regression algorithm for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    logging.info("Start the training the model")
    # Fit the algorithm to the data
    model = logit.fit(X_data, y_data)
    logging.info("Model done trainig!")
    # Write the trained model with name trainedmodel.pkl
    os.makedirs(model_path, exist_ok=True)
    pickle.dump(model, open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb'))
    logging.info("Save the model after training")

if __name__ == '__main__':
    logging.info("Running training.py script")
    train_model()
