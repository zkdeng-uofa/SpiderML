import pandas as pd
import sqlite3
import os
import subprocess
import argparse
from datasets import load_dataset
from huggingface_hub import login

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--numSpecies",
        type=str
    )
    parser.add_argument(
        "--numImages",
        type=str
    )
    return parser.parse_args()

def main():
    args = parse_args()

    numSpecies = args.numSpecies
    numImages = args.numImages
    datasetName = f'spiderTraining{numSpecies}-{numImages}'
    hf_token = "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"
    repo_name = "zkdeng/t5spiders"

    spidersDF = pd.read_csv("../data/csvs/spider_urls.csv")
    countDf = spidersDF.groupby('taxon_name').size().reset_index(name='count').sort_values(by='count', ascending=False)
    topSpecies = countDf[0:int(numSpecies)]
    filteredDf = spidersDF[spidersDF['taxon_name'].isin(topSpecies['taxon_name'])]
    spiderTrainDf = filteredDf.groupby('taxon_name').apply(lambda x: x.head(int(numImages))).reset_index(drop=True)

    spiderTrainDf.to_csv(f"../data/csvs/{datasetName}.csv")

    # Define the command as a list of arguments
    command = [
            "python", "ImgDownload.py",
            "--input_path", f"../data/csvs/{datasetName}.csv",
            "--output_folder", f"../data/imgs/{datasetName}",
            "--url_column", "photo_url",
            "--name_column", "taxon_name"
    ]

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and error, if any
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)
    print("Return Code:", result.returncode)

    login(token=hf_token)

    dataset = load_dataset("imagefolder", data_dir=f'../data/imgs/{datasetName}')

    dataset.push_to_hub(f'zkdeng/{datasetName}')

if __name__ == "__main__":
    main()


