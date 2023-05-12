from src.utils.all_utils import read_yaml, create_directory
import argparse
import pandas as pd
import os

def get_data(config_path):
    config = read_yaml(config_path)

    remote_data_path = config["data_source_url"]
    df = pd.read_csv(remote_data_path, sep=";")

    # save dataset in the local directory
    # create path to directory: artifacts/raw_local_dir/data.csv
    artifacts_dirName = config["artifactsLoc"]['artifacts_dirName']
    raw_data_dirName = config["artifactsLoc"]['raw_data_dirName']
    raw_data_fileName = config["artifactsLoc"]['raw_data_fileName']

    raw_local_dir_path = os.path.join(artifacts_dirName, raw_data_dirName)

    create_directory(dirs= [raw_local_dir_path])

    raw_local_file_path = os.path.join(raw_local_dir_path, raw_data_fileName)
    print(raw_local_file_path)
    
    df.to_csv(raw_local_file_path, sep=",", index=False)



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    get_data(config_path=parsed_args.config)
