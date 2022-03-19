from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class TextDataset(Dataset):
    """
    A CustomDataset to deliver text items
    """

    def __init__(
        self,
        tokeniser: PreTrainedTokenizerBase,
        data: pd.DataFrame,
        discourse_label_to_id: Optional[dict] = None,
        validate: bool = False,
        tokeniser_kwargs: Optional[dict] = None,
        label_sub_tokens: bool = True,
    ):
        self.data = data
        self.tokenise = tokeniser
        self.discourse_label_to_id = discourse_label_to_id
        self.tokenise_kwargs = tokeniser_kwargs or dict()
        self.validate = validate
        self.label_sub_tokens = label_sub_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: str) -> dict:
        """
        Get the text item with validation and labels

        :param index: id to get
        :return: (dict)
        """
        # The underlying item
        row = self.data.loc[index, :]
        text = row["text"].split()

        # Tokenise the text
        encoding = self.tokenise(text, is_split_into_words=True, **self.tokenise_kwargs)
        word_ids = encoding.word_ids()

        # Create labels if this isn't validation
        if not self.validate:
            word_labels = row["entities"]
            previous_word_idx = None
            label_ids = []
            for word_id in word_ids:
                # Don't include the token
                if word_id is None:
                    label_ids.append(-100)
                # This is a new word
                elif word_id != previous_word_idx:
                    label_ids.append(self.discourse_label_to_id[word_labels[word_id]])
                # A sub token
                else:
                    if self.label_sub_tokens:
                        label_ids.append(
                            self.discourse_label_to_id[word_labels[word_id]]
                        )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_id
            encoding["labels"] = label_ids

        # Get validation tensor if required
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.validate:
            validation_word_ids = [w if w is not None else -1 for w in word_ids]
            item["validation"] = torch.as_tensor(validation_word_ids)

        return item
