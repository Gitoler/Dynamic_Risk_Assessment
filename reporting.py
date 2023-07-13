"""
This script is to create reports of ML model
author: Ung Van Tuan
date: July 10th 2023
"""
import os
import json
import pickle
import logging
import pandas as pd
from datetime import datetime
from generate_pdf_report import PDF
from diagnostics import (
    model_predictions,
    execution_time,
    outdated_packages_list,
    missing_data,
    dataframe_summary)
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# Initialize logging
logging.basicConfig(filename='logs/reporting.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(name)s - %(levelname)s - %(message)s')

# Load the path variables from config.json
def get_config():
    with open('config.json','r') as f:
        config = json.load(f)
    dataset_csv_path = os.path.join(config['output_folder_path'])
    test_data_path = os.path.join(config['test_data_path'])
    deployment_path = os.path.join(config['prod_deployment_path'])
    return dataset_csv_path, test_data_path, deployment_path

def plot_confusion_matrix(y_test, y_predicted):
    cm = metrics.confusion_matrix(y_test, y_predicted)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g',cmap="crest")
    ax.set_title('Confusion Matrix', fontsize=20)
    ax.set_xlabel('\nPredicted Values')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticklabels(['0', '1'], fontsize = 20)
    ax.xaxis.tick_top()
    ax.set_ylabel('Actual Values',)
    ax.yaxis.set_ticklabels(['0', '1'], fontsize = 15)
    plt.savefig('images/confusion_matrix2.png')


def gen_report_pdf(ingestedfiles, latestscore,
                   df_lst, na_df_lst, timing,
                   dependencies_lst):
    """
    Function that generate the system report
    Input:
    ingestedfiles: str
    latestscore: str
    df_lst: List
    na_df_lst: List
    res_lst: List
    timing: Dictionary
    dependencies_lst: List
    """
    # Initialize the PDF 
    pdf = PDF()
    pdf.set_author('Ung Van Tuan')
    # Insert the data ingestion metadata from the txt file
    pdf.print_chapter(1, 'Ingested Data', ingestedfiles, "List of data ingested")
    pdf.chapter_subtitle("Summary Statistic")
    # Insert a table that contains Statistic the Summary
    with pdf.table() as table:
        for data_row in df_lst:
            row = table.row()
            for datum in data_row:
                row.cell(datum)
    pdf.chapter_subtitle("Missing data percentage")
    # Insert a table that contains the Missing data percentage
    with pdf.table() as na_table:
        for na_data_row in na_df_lst:
            na_row = na_table.row()
            for na_datum in na_data_row:
                na_row.cell(na_datum)
    # Insert the trained model evaluation from the text file
    pdf.print_chapter(2, 'Trained model evaluation on the test data', latestscore, "Latest Score")
    pdf.chapter_subtitle("Confusion Matrix")
    # Add the confusion matrix image
    pdf.image_("images/confusion_matrix2.png", 15, 90, 170)
    # Add the feature importance and the permutation importance images
    pdf.add_page()
    pdf.chapter_title(3, "Diagnostics for execution time and dependencies")
    pdf.chapter_subtitle("Ingestion and Training execution times")
    # insrt the time execution
    for key, value in timing.items():
        pdf.add_text(f"{key} = {value}")
    pdf.chapter_subtitle("Outdated Dependencies")
    # Add a table with the outdated libraries
    with pdf.table() as dep_table:
        for dep_data_row in dependencies_lst:
            dep_row = dep_table.row()
            for dep_datum in dep_data_row:
                dep_row.cell(dep_datum)
    day = str(datetime.now().strftime("%Y_%m_%d_%H-%M"))
    # Generate the pdf file
    pdf.output(f'report/system_report{day}.pdf', 'F')

def summary_statistic(stat_dict):
    """
    Function that convert the statistic summary 
    from dict to list of tuple
    Input:
    stat_dict: dict
    Output: List
    """
    stat_df = pd.DataFrame.from_dict(stat_dict)
    # Transpose the dataframe
    stat_df = stat_df.T
    # Add a new column into the dataframe
    stat_df.insert(loc = 0,
                   column = 'Column Name',
                   value = stat_df.index)
    df_str = stat_df.astype(str)
    # Get a list of tuple
    stat_lst = list(df_str.to_records(index=False))
    # Insert the column name value
    stat_lst.insert(0, tuple(df_str.keys()))
    return stat_lst

def missing_data_list(missing_list):
    """
    Function that convert a dict into list of tuple
    Input:
    missing_list: dict
    Output: List
    """
    na_df = pd.DataFrame.from_dict(missing_list)
    na_df_str = na_df.astype(str)
    na_df_lst = list(na_df_str.to_records(index=False))
    na_df_lst.insert(0, tuple(na_df.keys()))
    return na_df_lst

def dependencies_data_list(libraries_dict):
    """
    Function that convert a dict of outdated libraries into tuple
    Input:
    libraries_dict: dict
    Output:
    dependencies_lst: List
    """
    dependencies_df = pd.DataFrame.from_dict(libraries_dict)
    dependencies_str = dependencies_df.astype(str)
    dependencies_lst = list(dependencies_str.to_records(index=False))
    dependencies_lst.insert(0, tuple(dependencies_df.keys()))
    return dependencies_lst

def load_model(deployment_path):
    """
    Funtion to load the model
    Input:
    deployment_path: str
    Output:
    model: pickle
    """
    try:
        # collect deployed model
        with open(os.path.join(deployment_path, 'trainedmodel.pkl'), 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError as err:
        logging.error("Could not found the trainedmodel.pkl file")

    return model

# Function for reporting
def score_model():
    """
    Function that Calculate a confusion matrix using 
    the test data and the deployed model and generate a pdf report
    """
    _, test_data_path, deployment_path = get_config()
    ingestedfiles = os.path.join(deployment_path, 'ingestedfiles.txt')
    latestscore = os.path.join(deployment_path, 'latestscore.txt')
    try:
        dataset = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    except FileNotFoundError as err:
        logging.error("Error: Could not found the testdata.csv")
    y_test = dataset.pop('exited')
    X_test = dataset.drop(['corporation'], axis=1)
    y_pred = model_predictions(dataset)
    timing = execution_time()
    # Write the confusion matrix to the workspace
    plot_confusion_matrix(y_test,y_pred)
    # Load the model
    model = load_model(deployment_path)
    # Get the missing data
    missing_list = missing_data()
    sum_stat_dict = dataframe_summary()
    sum_stat_list = summary_statistic(sum_stat_dict)
    na_df_lst = missing_data_list(missing_list)
    libraries_dict = outdated_packages_list()
    dependencies_lst = dependencies_data_list(libraries_dict)
    # Generate the System report into a pdf
    gen_report_pdf(ingestedfiles, latestscore,
                   sum_stat_list, na_df_lst, timing,
                   dependencies_lst)

if __name__ == '__main__':
    score_model()