import gc

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers.modeling_utils import PreTrainedModel

MIN_BLOCK_SIZE = 7


class ModelHandler:
    """
    Wrapper to train and infer with an underlying ML model
    """

    def __init__(self, model: PreTrainedModel, optimizer: Optimizer, config: dict):
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def train(self, epoch: int, training_loader: DataLoader, max_batch_iter: int = 20):
        """
        For a given epoch, run training

        :param epoch: (int) of the training epoch
        :param training_loader: (DataLoader) to iterate through
        :param max_batch_iter: (int) Number of training iterations to stop at
        :return: (None) trains model
        """

        print(f"### Training epoch: {epoch + 1}")
        # Get learning rate for this epoch
        for g in self.optimizer.param_groups:
            g["lr"] = self.config["learning_rates"][epoch]
        lr = self.optimizer.param_groups[0]["lr"]
        print(f"### LR = {lr}\n")

        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # put model in training mode
        self.model.train()

        for idx, batch in enumerate(training_loader):
            # Only run to the specified iteration number
            if idx > max_batch_iter:
                break

            ids = batch["input_ids"].to(self.config["device"], dtype=torch.long)
            mask = batch["attention_mask"].to(self.config["device"], dtype=torch.long)
            labels = batch["labels"].to(self.config["device"], dtype=torch.long)

            # Get loss
            loss, tr_logits = self.model(
                input_ids=ids, attention_mask=mask, labels=labels, return_dict=False
            )

            tr_loss += float(loss.item())

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            # Log every 10th step
            if idx % 10 == 0:
                loss_step = tr_loss / nb_tr_steps
                print(f"Training loss after {idx:04d} training steps: {loss_step}")
                print("memory after predict", torch.cuda.memory_allocated(0) / 1e6)

            # compute training accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)

            active_logits = tr_logits.view(
                -1, self.model.num_labels
            )  # shape (batch_size * seq_len, num_labels)

            flattened_predictions = torch.argmax(
                active_logits, axis=1
            )  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            tmp_tr_accuracy = accuracy_score(
                labels.cpu().numpy(), predictions.cpu().numpy()
            )
            tr_accuracy += tmp_tr_accuracy

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=self.model.parameters(),
                max_norm=self.config["max_grad_norm"],
            )

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Memory management
            del loss
            del tr_logits
            del active_logits
            del ids
            del mask
            del labels
            gc.collect()
            torch.cuda.empty_cache()
            if idx % 10 == 0:
                print("memory after loop", torch.cuda.memory_allocated(0) / 1e6)

        epoch_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"Training loss epoch: {epoch_loss}")
        print(f"Training accuracy epoch: {tr_accuracy}")

    def predict(self, batch: dict, discourse_ids_to_labels: dict) -> list:
        """
        For a batch of items, make a prediction on the text parts

        :param batch: (dict) with keys
            ['input_ids', 'attention_mask', 'offset_mapping', 'labels']
        :param discourse_ids_to_labels: (dict) of discourse part id to label name
        :return: (list) of predictions
        """

        input_ids = batch["input_ids"].to(self.config["device"])
        attention_mask = batch["attention_mask"].to(self.config["device"])
        predict_tensor = self.model(
            input_ids, attention_mask=attention_mask, return_dict=False
        )
        # Get most likely discourse part id prediction
        predict_numpy = torch.argmax(predict_tensor[0], axis=-1).cpu().numpy()

        predictions = []
        # Go through each prediction
        for key, text_preds in enumerate(predict_numpy):
            # Get discourse label from id
            token_preds = [discourse_ids_to_labels[i] for i in text_preds]

            prediction = []
            # Only look for actual words
            word_ids = batch["validation"][key].numpy()

            previous_word_idx = -1

            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                # Get the full word not subwords
                elif word_idx != previous_word_idx:
                    prediction.append(token_preds[idx])
                    previous_word_idx = word_idx

            predictions.append(prediction)

        # Clean up for gpu management
        del input_ids
        del attention_mask
        del predict_tensor
        del predict_numpy
        gc.collect()
        torch.cuda.empty_cache()
        return predictions

    def get_all_predictions(
        self, df: pd.DataFrame, loader: DataLoader, discourse_ids_to_labels: dict
    ) -> pd.DataFrame:
        """
        Get all predictions by evaluations each row in the df

        :param df: (pd.DataFrame) to predict
        :param loader: (DataLoader) for loading datasets for prediction
        :param discourse_ids_to_labels: (dict) of discourse id to label
        :return: (pd.DataFrame) with predictions
        """

        # Model in evaluation mode
        self.model.eval()

        i = 0
        predictions = []
        for batch in loader:
            # Review memory
            if i % 100 == 0:
                print("batch", i)
                print("memory", torch.cuda.memory_allocated(0) / 1e6)
            i += 1
            # Get labels for batch
            labels = self.predict(batch, discourse_ids_to_labels)
            predictions.extend(labels)

        data_out = []
        for ix, row_id in enumerate(df.id):
            prediction = predictions[ix]
            # Go through the text string
            i = 0
            while i < len(prediction):
                cls = prediction[i]
                # This is not anything
                if cls == "0":
                    i += 1

                # Otherwise, clean Lead and Follow criteria
                else:
                    cls = cls.replace("L", "F")  # spans start with L

                end_of_block = i + 1
                # Get the end of this discourse block
                while (
                    end_of_block < len(prediction) and prediction[end_of_block] == cls
                ):
                    end_of_block += 1

                # As long as the block has min size and isn't nothing,
                # append the prediction
                if (
                    cls != "0"
                    and cls != ""
                    and end_of_block - i > MIN_BLOCK_SIZE
                ):
                    data_out.append(
                        (
                            row_id,
                            cls.replace("F-", ""),
                            " ".join(map(str, list(range(i, end_of_block)))),
                        )
                    )

                # Start of next block
                i = end_of_block

        df_out = pd.DataFrame(data_out)
        df_out.columns = ["id", "class", "predictionstring"]

        return df_out
