import os
import re

import gensim.downloader as api
import langdetect
import pandas as pd
import torch
import tqdm
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel, BertTokenizer

import B.visualising as vis
from A.data_acquisition import save_data


def text_preprocessing(dataset):
    # map the label to 0 and 1
    dataset_mapped = label_mapping(dataset)
    # remove different languages
    dataset_mapped["is_english"] = dataset_mapped["Review"].apply(text_english_chech)
    dataset_mapped = dataset_mapped[dataset_mapped["is_english"]].drop(
        columns=["is_english"]
    )

    # remove empty rows
    dataset_mapped = check_empty_row(dataset_mapped)

    # remove garbled text
    dataset_mapped = clean_garbled_text(dataset_mapped)

    # split the dataset into training, validation, and test sets
    trainset, valset, testset = construct_dataset(
        dataset_mapped, sample_size=15000
    )  # sample size is 15000 to each class
    # save the dataset
    data_folder = "./Datasets"
    save_data(trainset, data_folder, "original_trainset.csv")
    save_data(valset, data_folder, "original_valset.csv")
    save_data(testset, data_folder, "original_testset.csv")

    # using BERT
    # tokenize the dataset
    figure_folder = "./Figures"
    fig, _ = check_tokens_length(trainset["Review"])
    fig.savefig(
        os.path.join(figure_folder, "token_length_histogram.png"),
    )

    # embed the dataset
    trainset_file_name = "trainset"
    valset_file_name = "valset"
    testset_file_name = "testset"
    tokenization_embedding(
        trainset, data_folder, trainset_file_name
    )  # save the embeddings
    print("Embeddings saved for trainset.")
    tokenization_embedding(valset, data_folder, valset_file_name)  # save the embeddings
    print("Embeddings saved for valset.")
    tokenization_embedding(
        testset, data_folder, testset_file_name
    )  # save the embeddings
    print("Embeddings saved for testset.")

    # # using word2vec
    # # tokenize the dataset
    # trainset["tokens"] = trainset["Review"].apply(tokenization_word2vev)
    # valset["tokens"] = valset["Review"].apply(tokenization_word2vev)
    # testset["tokens"] = testset["Review"].apply(tokenization_word2vev)

    # # embed the dataset
    # # download the word2vec model
    # print("Downloading the word2vec model...")
    # model = api.load("word2vec-google-news-300")
    # print("Word2vec model downloaded.")
    # # embed the dataset, shape -> (num_tokens, embedding_dim)
    # trainset_tensor_list = [
    #     pad_or_truncate(tensor=word2vector(tokens, model), max_length=60)
    #     for tokens in trainset["tokens"]
    # ]
    # valset_tensor_list = [
    #     pad_or_truncate(tensor=word2vector(tokens, model), max_length=60)
    #     for tokens in valset["tokens"]
    # ]
    # testset_tensor_list = [
    #     pad_or_truncate(tensor=word2vector(tokens, model), max_length=60)
    #     for tokens in testset["tokens"]
    # ]
    # # convert tensor shape to (n, max_length, embedding_dim)
    # trainset_tensor = torch.stack(trainset_tensor_list)
    # valset_tensor = torch.stack(valset_tensor_list)
    # testset_tensor = torch.stack(testset_tensor_list)

    # # labels, shape -> (n, )
    # trainset_labels = torch.tensor(trainset["Freshness"].tolist())
    # valset_labels = torch.tensor(valset["Freshness"].tolist())
    # testset_labels = torch.tensor(testset["Freshness"].tolist())

    # # construct the dataset
    # torch.save(
    #     (trainset_tensor, trainset_labels),
    #     os.path.join(data_folder, "word2_trainset_embeddings.pt"),
    # )
    # torch.save(
    #     (valset_tensor, valset_labels),
    #     os.path.join(data_folder, "word2_valset_embeddings.pt"),
    # )
    # torch.save(
    #     (testset_tensor, testset_labels),
    #     os.path.join(data_folder, "word2_testset_embeddings.pt"),
    # )

    # return (
    #     trainset_tensor,
    #     trainset_labels,
    #     valset_tensor,
    #     valset_labels,
    #     testset_tensor,
    #     testset_labels,
    # )


def text_english_chech(text):
    """
    Check if the text is in English.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is in English, False otherwise.
    """
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def label_mapping(dataset: pd.DataFrame):
    """
    Map the labels in the dataset to numerical values.

    Args:
        dataset (pd.DataFrame): The dataset containing the labels.

    Returns:
        pd.DataFrame: The dataset with mapped labels.
    """

    label_map = {
        "rotten": 0,
        "fresh": 1,
    }

    dataset["Freshness"] = dataset["Freshness"].map(label_map)
    return dataset


def check_empty_row(dataset: pd.DataFrame):
    empty_bool = dataset[["Review", "Freshness"]].isnull().any(axis=1)
    empty_num = empty_bool.sum()
    print(f"Number of empty rows: {empty_num}")
    if empty_num > 0:
        print("Empty rows detected. Removing them...")
        dataset = remove_empty_row(dataset)
        print("Empty rows removed.")
    else:
        print("No empty rows detected.")
    return dataset


def remove_empty_row(dataset: pd.DataFrame):
    """
    Remove empty rows from the dataset.

    Args:
        dataset (pd.DataFrame): The dataset to remove empty rows from.

    Returns:
        pd.DataFrame: The dataset with empty rows removed.
    """
    dataset = dataset.dropna(subset=["Review", "Freshness"])
    return dataset.reset_index(drop=True)


def construct_dataset(dataset: pd.DataFrame, sample_size: int):
    # Construct a balanced dataset with equal number of positive and negative samples
    df_positive = dataset[dataset["Freshness"] == 1].sample(
        n=sample_size, random_state=711
    )
    df_negative = dataset[dataset["Freshness"] == 0].sample(
        n=sample_size, random_state=711
    )

    df_final = (
        pd.concat([df_positive, df_negative], ignore_index=True)
        .sample(frac=1, random_state=711)
        .reset_index(drop=True)
    )

    # Split the dataset into training, validation, and test sets
    x = df_final["Review"]
    y = df_final["Freshness"]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=711,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_val,
        y_val,
        test_size=0.5,
        random_state=711,
    )

    trainset = pd.DataFrame({"Review": x_train, "Freshness": y_train})
    valset = pd.DataFrame({"Review": x_val, "Freshness": y_val})
    testset = pd.DataFrame({"Review": x_test, "Freshness": y_test})
    return trainset, valset, testset


def clean_garbled_text(dataset):
    """
    Remove garbled text from the dataset.
    Args:
        dataset (pd.DataFrame): The dataset to clean.
    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    df_cleaned = dataset.copy()

    def remove_garbled_text(text):
        # Remove non-ASCII characters
        return all(ord(char) < 128 for char in str(text))

    mask = df_cleaned["Review"].apply(remove_garbled_text)
    df_cleaned = df_cleaned[mask].reset_index(drop=True)
    return df_cleaned


def check_tokens_length(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lengths = [len(tokenizer.tokenize(text)) for text in texts]
    fig, ax = vis.plot_hist(lengths)
    return fig, ax


def tokenization_embedding(dataset: pd.DataFrame, folder_path: str, file_name: str):
    texts = dataset["Review"].tolist()
    labels = dataset["Freshness"].tolist()

    # Load the BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    max_length = 60

    all_embeddings = []
    all_labels = []
    all_attention_masks = []

    with torch.no_grad():
        for text, label in tqdm.tqdm(zip(texts, labels), total=len(texts)):
            # Tokenize the text
            inputs = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Get the BERT embeddings
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.squeeze(
                0
            )  # (num_tokens, hidden_size)
            attention_mask = inputs["attention_mask"].squeeze(0)  # (num_tokens)

            all_embeddings.append(embeddings)
            all_labels.append(label)
            all_attention_masks.append(attention_mask)
    torch.save(
        all_embeddings,
        os.path.join(folder_path, file_name + "_embeddings.pt"),
    )
    torch.save(
        all_labels,
        os.path.join(folder_path, file_name + "_labels.pt"),
    )
    torch.save(
        all_attention_masks,
        os.path.join(folder_path, file_name + "_attention_masks.pt"),
    )


def tokenization_word2vev(text):
    # lowercase the text
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-z\s]", "", text)
    text = text.strip().split()
    return text


def word2vector(tokens, model):

    word_vectors = []
    for word in tokens:
        if word in model:
            vec = torch.tensor(model[word]).float()
            word_vectors.append(vec)

    if word_vectors:
        # stack the word vectors into a tensor
        return torch.stack(word_vectors)

    else:
        # return a zero tensor if no word vectors are found
        return torch.empty((0, 300))


def pad_or_truncate(tensor, max_length, emdedding_dim=300):
    """
    Pad or truncate a tensor to a specified maximum length.

    Args:
        tensor (torch.Tensor): The tensor to pad or truncate.
        max_length (int): The maximum length to pad or truncate to.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    if tensor.size(0) > max_length:
        return tensor[:max_length]
    elif tensor.size(0) < max_length:
        padding_len = max_length - tensor.size(0)
        padding = torch.zeros(padding_len, emdedding_dim)
        return torch.cat((tensor, padding), dim=0)
    else:
        return tensor
