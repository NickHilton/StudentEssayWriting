import os
from typing import Dict
import pandas as pd
from autocorrect import Speller


def process_file(filepath: str) -> str:
    """
    Process a file and read the lines from it
    :param filepath: (str) to read
    :return: (str) file contents
    """
    with open(filepath) as f:
        return f.read()


def process_dir(dirpath: str) -> pd.DataFrame:
    """
    Process all files in a directory, return a dataframe of items with columns
        ['id', 'text', 'textlength']
    :param dirpath: (str) directory to process
    :return: (pd.DataFrame) with columns ['id', 'text']
    """
    spell = Speller("en", fast=True)
    id_to_text: Dict[str, str] = dict()
    for filename in os.listdir(dirpath):
        f = os.path.join(dirpath, filename)
        text = spell(process_file(f))
        text_id = filename.split(".txt")[0]
        id_to_text[text_id] = text

    df = pd.DataFrame(id_to_text.items(), columns=["id", "text"])
    df["text_length"] = df["text"].apply(lambda text_item: len(text_item.split()))
    return df


def process_train_df(train_df_path: str) -> pd.DataFrame:
    """
    Get and clean the train df containing file ids and their text parts

    :param train_df_path: (str) to process
    :return: (pd.DataFrame) of data
    """

    df = pd.read_csv(train_df_path)
    for col in ["discourse_id", "discourse_start", "discourse_end"]:
        df[col] = df[col].astype(int)

    return df
