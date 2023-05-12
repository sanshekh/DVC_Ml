from src.utils.all_utils import read_yaml, create_directory, save_local_df
import argparse
import pandas as pd
import os
from sklearn.linear_model import ElasticNet
import joblib


def train(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)


    artifacts_dirName = config["artifactsLoc"]['artifacts_dirName']
    split_data_dirName = config["artifactsLoc"]["split_data_dirName"]


    train_data_filename = config["artifactsLoc"]["train_fileName"]


    train_data_path = os.path.join(artifacts_dirName, split_data_dirName, train_data_filename)
    
    train_data = pd.read_csv(train_data_path)

    train_y = train_data["quality"]
    train_x = train_data.drop("quality", axis=1)

    alpha = params["model_params"]["ElasticNet"]["alpha"]
    l1_ratio = params["model_params"]["ElasticNet"]["l1_ratio"]
    random_state = params["base_params"]["random_state"]


    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)

    model_dirName = config["artifactsLoc"]["model_dirName"]
    model_fileName = config["artifactsLoc"]["model_fileName"]

    model_dir = os.path.join(artifacts_dirName, model_dirName)

    create_directory([model_dir])

    model_path = os.path.join(model_dir, model_fileName)


    joblib.dump(lr, model_path)

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    train(config_path=parsed_args.config, params_path=parsed_args.params)
