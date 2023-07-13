"""
This script is to get model scoring
author: Ung Van Tuan
date: July 10th 2023
"""
import logging
import pandas as pd
import pickle
import os
from sklearn import metrics
import json

# Initial logging
logging.basicConfig(filename='logs/scoring.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Get the path variables from config.json
def get_config():
    with open('config.json','r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])
    test_data_path = os.path.join(config['test_data_path'])
    return model_path, test_data_path


# Create function for the model score
def score_model():
    """
    Function to accomplish model score
    """
    # this function to take a trained model, load test data to have the score

    model_path, test_data_path = get_config()
    logging.info("Loading testdata.csv")
    test_df = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    y_test = test_df.pop('exited')
    X_test = test_df.drop(['corporation'], axis=1)

    # Load the model from trainedmodel.pkl file
    logging.info("Loading trained model")
    model = pickle.load(open(os.path.join(model_path, 'trainedmodel.pkl'), 'rb'))

    # Evaluate the model
    y_pred = model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)
    f1_score = round(f1_score, 2)

    # Save the evaluation score into a file
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as file:
        file.write(f"f1_score = {f1_score}")
    logging.info("Evaluation scores saved")
    print(f1_score)

if __name__ == '__main__':
    logging.info("Executing scoring.py script")
    score_model()