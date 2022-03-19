import os
import shutil
from typing import Optional

from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification

# The underlying tokeniser model
MODEL_NAME = os.environ.get("MODEL_NAME", "google/bigbird-roberta-base")
# Where the model is being saved
MODEL_PATH = os.environ.get("MODEL_PATH", "tokeniser")


def clean_model_dir(model_path: str = MODEL_PATH) -> bool:
    """
    Clean the tokeniser dir in preparation for saving, deleting anything at the
    tokeniser path if any exists

    :param model_path: (str) tokeniser path to clean
    :return: (bool) True if successful
    """
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.mkdir(model_path)
    return True


def get_model_from_external(
    model_name: str = MODEL_NAME,
    model_path: str = MODEL_PATH,
    config_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
):
    """
    Download a given tokeniser for tokenisation and save to the model_path
    Useful for saving a tokeniser to a local file

    :param model_name: (str) to download
    :param model_path: (str) to save to
    :param config_kwargs: (dict or None) to set on the config
    :param model_kwargs: (dict or None) to pass to the tokeniser
    :return: (bool) True if tokeniser saved successfully
    """
    # Check tokeniser not already there
    # if os.path.exists(model_path):
    #     raise ValueError(f"Model already exists at {model_path}")
    os.mkdir(model_path)
    config = AutoConfig.from_pretrained(model_name)
    # Set any config kwargs
    if config_kwargs:
        for attr, val in config_kwargs.items():
            setattr(config, attr, val)
    # Save config
    config.save_pretrained(model_path)

    # Get tokeniser
    if not model_kwargs:
        model_kwargs = dict()
    tokeniser = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
    tokeniser.save_pretrained(model_path)

    # Get final tokeniser
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)
    model.save_pretrained(model_path)
    return True
