import pandas as pd
from typing import List


def get_prediction_string(
    text_id: int, text_length: int, train_df: pd.DataFrame
) -> List[str]:
    """
    For a given text_id, get the labelled predictions, converting the predictions
    into 'Lead' and 'Follow' elements and returning an array of discourse parts,
    corresponding to the position of each word in the original text

    :param text_id: (str) text_id
    :param text_length: (int) length of text
    :param train_df: (pd.DataFrame) labels for data
    :return: (list(str)) of discourse parts
    """
    # Get data for this text_id
    id_df = train_df.query(f"id == '{text_id}'")

    # Set up return array -> 0 == no prediction
    predictions = ["0"] * text_length

    if id_df.shape[1] > 0:
        for _, row in id_df.iterrows():
            # For each discourse part
            discourse_type = row["discourse_type"]
            lead = True
            for pos in map(int, row["predictionstring"].split()):
                # Record a leading word of the discourse part
                if lead:
                    predictions[pos] = f"L-{discourse_type}"
                    lead = False
                else:  # is a follower item in the section
                    predictions[pos] = f"F-{discourse_type}"
    return predictions


def tokenise(train_df: pd.DataFrame, texts_df: pd.DataFrame):
    """
    Get the prediction parts for a text item, updating the input_df
    :param train_df: (pd.DataFrame) of training data
    :param texts_df: (pd.DataFrame) of texts and ids
    :return: (pd.DataFrame)
    """
    texts_df["entities"] = texts_df.apply(
        lambda row: get_prediction_string(row["id"], row["text_length"], train_df),
        axis=1,
    )
    return texts_df
