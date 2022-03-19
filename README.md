# Student Essay Text Part Classification

## Background
This repo was developed to tackle [this Kaggle competition](https://www.kaggle.com/c/feedback-prize-2021/overview) which takes essays written by students (Grade 6-12) and attempts to classify the parts of the essay (i.e. Lead, Position, Evidence, Claim)

It uses a MLM approach, taking the [huggingface extension of BigBird](https://huggingface.co/google/bigbird-roberta-base) and applying that to the essays.

## Running an experiment

1. Clone the repository
2. Create a virtual environment and install the project requirements
```bash
pip install -r requirements.txt
```
3. Load up the `student_writing_part_detection.ipynb` notebook and test things out!