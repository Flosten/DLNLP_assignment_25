"""
This module contains the dataloaders creation, model architecture and training functions
for the text classification task, as well as ablation studies.

functions:
- `set_seed`: Set the random seed for reproducibility.
- `prepare_dataloader_full_pipleine`: Prepare the dataloader for the full pipeline.
- `prepare_dataloader`: Prepare the dataloader by loading the dataset from the folder path.
- `preprocess_word2vec`: Preprocess the word2vec features and labels.
- `BaselineModel`: A simple baseline model for text classification.
- `SimpleLSTM`: A simple LSTM model for text classification.
- `CNNLSTM`: A CNN + LSTM model for text classification.
- `CNNLSTMAttn`: A CNN + LSTM + Attention model for text classification.
- `train_model`: Train the model.
- `train_model_attn`: Train the model with attention mechanism.
- `train_model_lr_decay`: Train the model with learning rate decay strategy.
- `evaluate_model`: Evaluate the model on the test set.
- `evaluate_model_attn`: Evaluate the model with attention mechanism on the test set.
"""

import os
import random

import numpy as np

# import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

import B.visualising as vis


def set_seed(seed=711):
    """
    Set the random seed for reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Torch multiprocessing
    os.environ["PYTHONHASHSEED"] = str(seed)


def prepare_dataloader_full_pipleine(
    embeddings,
    attention_masks,
    labels,
    batch_size,
):
    """
    This function contains the full pipeline of the preprocessing of the data.

    Args:
        embeddings (torch.Tensor): The embeddings of the data.
        attention_masks (torch.Tensor): The attention masks of the data.
        labels (torch.Tensor): The labels of the data.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: The DataLoader for the data.
    """

    def load_dataset(
        dataset_embeddings,
        dataset_attention_masks,
        dataset_labels,
    ):
        embeddings = dataset_embeddings
        attention_masks = dataset_attention_masks
        labels = dataset_labels

        class moviedataset(Dataset):
            def __init__(self, embeddings, attention_masks, labels):
                self.embeddings = embeddings
                self.attention_masks = attention_masks
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.embeddings[idx], self.attention_masks[idx], self.labels[idx]

        return moviedataset(embeddings, attention_masks, labels)

    dataset = load_dataset(embeddings, attention_masks, labels)

    dataset_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    return dataset_loader


def prepare_dataloader(folder_path, batch_size):
    """
    This function is used to load the dataset from the folder path
    and prepare the dataloader for the model.

    Args:
        folder_path (str): The path to the folder containing the dataset.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        tuple: A tuple containing the trainset_loader, valset_loader, and testset_loader.
    """

    def load_dataset(folder_path, file_type):
        """
        Load the dataset from the folder path.

        Args:
            folder_path (str): The path to the folder containing the dataset.
            file_type (str): The type of the dataset (trainset, valset, testset).

        Returns:
            moviedataset: The dataset object.
        """
        embedding_file = f"{file_type}_embeddings.pt"
        embeddings_path = os.path.join(folder_path, embedding_file)
        attention_mask_file = f"{file_type}_attention_masks.pt"
        attention_masks_path = os.path.join(folder_path, attention_mask_file)
        labels_file = f"{file_type}_labels.pt"
        labels_path = os.path.join(folder_path, labels_file)
        embeddings = torch.load(embeddings_path)
        attention_masks = torch.load(attention_masks_path)
        labels = torch.load(labels_path)

        class moviedataset(Dataset):
            """
            A dataset class for the movie reviews dataset.
            """

            def __init__(self, embeddings, attention_masks, labels):
                self.embeddings = embeddings
                self.attention_masks = attention_masks
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.embeddings[idx], self.attention_masks[idx], self.labels[idx]

        return moviedataset(embeddings, attention_masks, labels)

    train_dataset = load_dataset(folder_path, "trainset")
    val_dataset = load_dataset(folder_path, "valset")
    test_dataset = load_dataset(folder_path, "testset")

    trainset_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    valset_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=True,
    )
    testset_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=True,
    )
    return trainset_loader, valset_loader, testset_loader


def preprocess_word2vec(features, label, batch_size, shuffle):
    """
    Create a DataLoader for the word2vec features and labels.

    Args:
        features (torch.Tensor): The word2vec features.
        label (torch.Tensor): The labels.
        batch_size (int): The batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: The DataLoader for the word2vec features and labels.
    """
    dataset = TensorDataset(features, label)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )
    return loader


class BaselineModel(nn.Module):
    """
    A simple baseline model for text classification.
    """

    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x, mask):
        """
        Forward pass of the baseline model.

        Args:
            x (torch.Tensor): The input data.
            mask (torch.Tensor): The attention mask from BERT.

        Returns:
            torch.Tensor: The output of the model.
        """
        mask = mask.unsqueeze(-1)  # [batch, seq_len] -> [batch, seq_len, 1]
        x = x * mask  # [batch, seq_len, 768]

        sum_x = x.sum(dim=1)  # [batch, 768]
        avg_x = sum_x / mask.sum(dim=1)  # [batch, 768]
        # avg_x = x.mean(dim=1)  # [batch, 768]

        out = self.dropout(avg_x)  # [batch, 768]
        out = self.classifier(out)  # [batch, num_classes]
        return out  # [batch, num_classes]


class SimpleLSTM(nn.Module):
    """
    A simple LSTM model for text classification.
    """

    def __init__(self, hidden_dim=50, num_classes=2):
        super().__init__()
        # self.project = nn.Linear(768, 30)
        # self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_dim,
            batch_first=True,
            # dropout=0.3,
            num_layers=1,
        )
        self.dropout = nn.Dropout(0.6)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): The input data.
            mask (torch.Tensor): The attention mask from BERT.

        Returns:
            torch.Tensor: The output of the model.
        """
        # x: [batch, seq_len, 768]
        # x = self.project(x)  # → [batch, seq_len, 128]
        # x = self.dropout(x)

        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_n = self.dropout(h_n)  # [num_layers * num_directions, batch, hidden_dim]
        out = h_n[-1]  # [batch, hidden_dim]
        # out = self.dropout(out)
        return self.classifier(out)


class CNNLSTM(nn.Module):
    """
    This is a CNN + LSTM model for text classification.
    """

    def __init__(
        self,
        input_size=768,
        # seq_len=60,
        hidden_size=50,
        num_classes=2,
        cnn_out=128,
        kernel_size=5,
    ):
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out,
            kernel_size=kernel_size,
            padding=2,
        )
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            batch_first=True,
            # dropout=0.3,
            bidirectional=False,
        )
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.6)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, mask):
        """
        Forward pass of the CNN + LSTM model.

        Args:
            x (torch.Tensor): The input data.
            mask (torch.Tensor): The attention mask from BERT.

        Returns:
            torch.Tensor: The output of the model.
        """
        # x: [batch, seq_len, input_size] -> [batch, input_size, seq_len]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # [batch, cnn_out, seq_len]
        x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, cnn_out]
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )  # [batch, seq_len, cnn_out]
        _, (h_n, _) = self.lstm(packed)  # [batch, seq_len, hidden_size]
        h_n = self.dropout2(h_n)  # [num_layers * num_directions, batch, hidden_size]
        h_n = h_n[-1]  # [batch, hidden_size]
        output = self.fc(h_n)  # [batch, num_classes]
        return output  # [batch, num_classes]


class CNNLSTMAttn(nn.Module):
    """
    This model combines CNN, LSTM, and attention mechanism for text classification.
    """

    def __init__(
        self,
        input_size=768,
        # seq_len=60,
        hidden_size=50,
        num_classes=2,
        cnn_out=128,
        kernel_size=5,
    ):
        super().__init__()
        self.cnn = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out,
            kernel_size=kernel_size,
            padding=2,
        )  # 1D convolution, shape [batch, input_size, seq_len] -> [batch, cnn_out, seq_len]
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=hidden_size,
            batch_first=True,
            # dropout=0.5,
            num_layers=1,
            bidirectional=False,
        )  # LSTM layer, shape [batch, seq_len, cnn_out] -> [batch, seq_len, hidden_size]
        self.dropout1 = nn.Dropout(0.4)  # dropout layer
        self.dropout2 = nn.Dropout(0.6)  # dropout layer
        # attention layer
        self.attention = nn.Linear(hidden_size, 1)
        # fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def attention_mechanism(self, lstm_out, mask):
        """
        Attention mechanism to compute the context vector and attention weights.

        Args:
            lstm_out (torch.Tensor): The output from the LSTM layer.
            mask (torch.Tensor): The attention mask from BERT.

        Returns:
            torch.Tensor: The context vector.
            torch.Tensor: The attention weights.
        """
        attn_weights = self.attention(lstm_out).squeeze(-1)  # [batch, seq_len]
        mask = mask.to(dtype=torch.bool)

        attn_weights[~mask] = float("-inf")  # Set masked positions to -inf
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch, seq_len]
        attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len, 1]
        context_vector = torch.sum(
            attn_weights * lstm_out, dim=1
        )  # [batch, hidden_size]
        return context_vector, attn_weights

    def forward(self, x, mask):
        """
        Forward pass of the CNN + LSTM + Attention model.

        Args:
            x (torch.Tensor): The input data.
            mask (torch.Tensor): The attention mask.

        Returns:
            torch.Tensor: The output of the model.
            torch.Tensor: The attention scores.
        """
        # x: [batch, seq_len, input_size] -> [batch, input_size, seq_len]
        x = x.permute(0, 2, 1)  # [batch, input_size, seq_len]
        x = self.cnn(x)  # [batch, cnn_out, seq_len]
        # x = self.dropout1(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, cnn_out]
        lstm_output, _ = self.lstm(x)  # [batch, seq_len, hidden_size]
        # lstm_output = self.dropout2(lstm_output)
        attention_out, attention_score = self.attention_mechanism(
            lstm_output, mask
        )  # [batch, hidden_size]
        # attention_out = self.dropout1(attention_out)
        output = self.fc(attention_out)  # [batch, num_classes]
        return output, attention_score  # [batch, num_classes], [batch, seq_len, 1]


# def train_model(
#     model,
#     trainset_loader,
#     valset_loader,
#     optimizer,
#     criterion,
#     num_epochs=10,
# ):
#     train_loss = []
#     val_loss = []
#     # train_acc = []
#     # val_acc = []

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         train_accuracy = 0
#         for input, attention_mask, label in tqdm.tqdm(
#             trainset_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
#         ):
#             optimizer.zero_grad()
#             pred = model(input, attention_mask)
#             loss = criterion(pred, label)
#             _, predicted = torch.max(pred, 1)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         train_loss.append(total_loss / len(trainset_loader))
#         train_accuracy = accuracy_score(label.cpu().numpy(), predicted.cpu().numpy())
#         # train_acc.append(train_accuracy)

#         model.eval()
#         epoch_val_loss = 0
#         with torch.no_grad():
#             for input, attention_mask, label in valset_loader:
#                 pred = model(input, attention_mask)
#                 _, predicted = torch.max(pred, 1)
#                 loss = criterion(pred, label)
#                 epoch_val_loss += loss.item()
#         val_loss.append(epoch_val_loss / len(valset_loader))
#         val_accuracy = accuracy_score(label.cpu().numpy(), predicted.cpu().numpy())
#         # val_acc.append(val_accuracy)
#         print(
#             f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}"
#         )
#         print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")
#     fig, ax = vis.plot_learning_curve(train_loss, val_loss)

#     return model, fig, ax


def train_model(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    num_epochs=10,
):
    """
    This function is used for baseline model training and ablation study.

    Args:
        model (nn.Module): The model to train.
        trainset_loader (DataLoader): The DataLoader for the training set.
        valset_loader (DataLoader): The DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        criterion (nn.Module): The loss function.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the trained model, the figure, axis, and the training accuracy.
    """
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for x, attention_mask, labels in tqdm.tqdm(
            trainset_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):

            optimizer.zero_grad()
            outputs = model(x, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu())

        train_loss.append(total_loss / len(trainset_loader))
        train_accuracy = accuracy_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
        )
        train_acc.append(train_accuracy)

        # Validation
        model.eval()
        val_total_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, attention_mask, labels in valset_loader:

                outputs = model(x, attention_mask)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                val_preds.append(predicted.detach().cpu())
                val_labels.append(labels.detach().cpu())

        val_loss.append(val_total_loss / len(valset_loader))
        val_accuracy = accuracy_score(
            torch.cat(val_labels).numpy(), torch.cat(val_preds).numpy()
        )
        val_acc.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}"
        )
        print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    fig, ax = vis.plot_learning_curve(train_loss, val_loss)

    return model, fig, ax, train_accuracy


def train_model_attn(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    num_epochs=10,
):
    """
    This function is used for training the model with attention mechanism.

    Args:
        model (nn.Module): The model to train.
        trainset_loader (DataLoader): The DataLoader for the training set.
        valset_loader (DataLoader): The DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        criterion (nn.Module): The loss function.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the trained model, the figure, axis, and the training accuracy.
    """
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for x, attention_mask, labels in tqdm.tqdm(
            trainset_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):

            optimizer.zero_grad()
            outputs, _ = model(x, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu())

        train_loss.append(total_loss / len(trainset_loader))
        train_accuracy = accuracy_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
        )
        train_acc.append(train_accuracy)

        # Validation
        model.eval()
        val_total_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, attention_mask, labels in valset_loader:

                outputs, _ = model(x, attention_mask)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                val_preds.append(predicted.detach().cpu())
                val_labels.append(labels.detach().cpu())

        val_loss.append(val_total_loss / len(valset_loader))
        val_accuracy = accuracy_score(
            torch.cat(val_labels).numpy(), torch.cat(val_preds).numpy()
        )
        val_acc.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}"
        )
        print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    fig, ax = vis.plot_learning_curve(train_loss, val_loss)

    return model, fig, ax, train_accuracy


def train_model_lr_decay(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    criterion,
    scheduler,
    num_epochs=10,
):
    """
    This function is used for training the model with learning rate decay strategy.

    Args:
        model (nn.Module): The model to train.
        trainset_loader (DataLoader): The DataLoader for the training set.
        valset_loader (DataLoader): The DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        criterion (nn.Module): The loss function.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the trained model, the figure, axis.
    """
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for x, attention_mask, labels in tqdm.tqdm(
            trainset_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):

            optimizer.zero_grad()
            outputs = model(x, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu())

        train_loss.append(total_loss / len(trainset_loader))
        train_accuracy = accuracy_score(
            torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()
        )
        train_acc.append(train_accuracy)
        scheduler.step()

        # Validation
        model.eval()
        val_total_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for x, attention_mask, labels in valset_loader:

                outputs = model(x, attention_mask)
                loss = criterion(outputs, labels)
                val_total_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                val_preds.append(predicted.detach().cpu())
                val_labels.append(labels.detach().cpu())

        val_loss.append(val_total_loss / len(valset_loader))
        val_accuracy = accuracy_score(
            torch.cat(val_labels).numpy(), torch.cat(val_preds).numpy()
        )
        val_acc.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}"
        )
        print(f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

    fig, ax = vis.plot_learning_curve(train_loss, val_loss)

    return model, fig, ax


def evaluate_model(model, test_loader):
    """
    This function is used to evaluate the model on the test set.
    To be specific, it is for the baseline model and the ablation study.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The DataLoader for the test set.

    Returns:
        tuple: A tuple containing the accuracy, F1 score, figure, and axis.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, mask, y in test_loader:
            logits = model(x, mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    _, fig, ax = vis.evalute_the_preformance(all_preds, all_labels)
    return accuracy, f1, fig, ax


def evaluate_model_attn(model, test_loader):
    """
    This function is used to evaluate the model with attention mechanism on the test set.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): The DataLoader for the test set.

    Returns:
        tuple: A tuple containing the accuracy, F1 score, figure, axis, attention figure, and axis.
    """

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_attention_scores = []
    all_masks = []

    with torch.no_grad():
        for x, mask, y in test_loader:
            logits, attention_score = model(x, mask)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_attention_scores.extend(attention_score.cpu().numpy())
            all_masks.extend(mask.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    _, fig, ax = vis.evalute_the_preformance(all_preds, all_labels)
    fig_attn, ax_attn = vis.attention_score_vis(
        all_attention_scores,
        all_masks,
    )
    return accuracy, f1, fig, ax, fig_attn, ax_attn
