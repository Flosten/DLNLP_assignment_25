"""
This is the main file for the sentiment analysis project.
"""

import os

# import numpy as np
# import pandas as pd
# import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import A.data_acquisition as da
import A.data_preprocessing as dp
import B.modelling as mdl

# from torch.optim.lr_scheduler import StepLR


def sentiment_analysis():
    """
    Main function to run the sentiment analysis pipeline.
    It includes movie review data acquisition, preprocessing, model training,
    evaluation, and visualization.
    """
    # set the folder path and file name
    data_folder = os.path.join(".", "Datasets")
    figure_folder = os.path.join(".", "Figures")
    original_dataset = "dataset_select.csv"
    trainset_name = "original_trainset.csv"
    valset_name = "original_valset.csv"
    testset_name = "original_testset.csv"
    # embedded dataset name
    embedded_trainset_name = "trainset"
    embedded_valset_name = "valset"
    embedded_testset_name = "testset"

    # set the hyperparameters
    batch_size = 32
    num_epochs = 13
    # learning_rate_cnnlstmatn = 2e-5
    learning_rate_baseline = 1e-4
    learning_rate_ablation = 1.5e-5
    # num_classes = 2

    # check if the folders exist, if not create them
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)

    # # Load the dataset
    # # all the data preprocessing pipeline
    # data_select = da.load_data(data_folder, original_dataset)
    # (
    #     trainset_embeddings,
    #     trainset_labels,
    #     trainset_masks,
    #     valset_embeddings,
    #     valset_labels,
    #     valset_masks,
    #     testset_embeddings,
    #     testset_labels,
    #     testset_masks,
    # ) = dp.text_preprocessing(data_select)

    # -------------------------------------------------
    # load the preprocessed trainset, valset, and testset
    trainset = da.load_data(data_folder, trainset_name)
    valset = da.load_data(data_folder, valset_name)
    testset = da.load_data(data_folder, testset_name)

    # tokenize the data
    # check the length of the tokens
    fig_token, _ = dp.check_tokens_length(trainset["Review"])
    fig_token.savefig(os.path.join(figure_folder, "trainset_tokens_length.png"))

    # tokenize and embed the data
    trainset_embeddings, trainset_labels, trainset_attention_masks = (
        dp.tokenization_embedding(trainset, data_folder, embedded_trainset_name)
    )
    valset_embeddings, valset_labels, valset_attention_masks = (
        dp.tokenization_embedding(valset, data_folder, embedded_valset_name)
    )
    testset_embeddings, testset_labels, testset_attention_masks = (
        dp.tokenization_embedding(testset, data_folder, embedded_testset_name)
    )

    # set seed for reproducibility
    mdl.set_seed(711)

    # build the dataloaders
    train_loader = mdl.prepare_dataloader_full_pipleine(
        trainset_embeddings,
        trainset_attention_masks,
        trainset_labels,
        batch_size,
    )
    val_loader = mdl.prepare_dataloader_full_pipleine(
        valset_embeddings,
        valset_attention_masks,
        valset_labels,
        batch_size,
    )
    test_loader = mdl.prepare_dataloader_full_pipleine(
        testset_embeddings,
        testset_attention_masks,
        testset_labels,
        batch_size,
    )

    # # --------------------------------------------------
    # train_loader, val_loader, test_loader = mdl.prepare_dataloader(
    #     data_folder,
    #     batch_size,
    # )

    # --------------------------------------------------

    # build the baseline model
    print("Baseline model training")
    baseline = mdl.BaselineModel()
    optimizer_baseline = Adam(baseline.parameters(), lr=learning_rate_baseline)
    criterion_baseline = CrossEntropyLoss()
    baseline_trained, fig_baseline_lr, _, baseline_train_acc = mdl.train_model(
        baseline,
        train_loader,
        val_loader,
        optimizer_baseline,
        criterion_baseline,
        num_epochs,
    )
    # save the baseline model learning curve
    fig_baseline_lr.savefig(
        os.path.join(figure_folder, "baseline_model_learning_curve.png")
    )
    # test the baseline model
    acc_baseline, f1_baseline, fig_baseline, _ = mdl.evaluate_model(
        baseline_trained,
        test_loader,
    )

    # save the baseline model performance
    fig_baseline.savefig(os.path.join(figure_folder, "baseline_model_performance.png"))

    # CNN + LSTM + Attention without learning rate decay
    print("CNN + LSTM + Attention model training")
    cnn_lstm_attention_1 = mdl.CNNLSTMAttn()
    learning_rate_cnnlstmatn_1 = 1.5e-5
    optimizer_cnn_lstm_attention_1 = Adam(
        cnn_lstm_attention_1.parameters(),
        lr=learning_rate_cnnlstmatn_1,
        weight_decay=1e-3,
    )
    criterion_cnn_lstm_attention_1 = CrossEntropyLoss()
    cnn_lstm_attention_1_trained, fig_cnn_lstm_attention_1_lr, _, mymodel_train_acc = (
        mdl.train_model_attn(
            cnn_lstm_attention_1,
            train_loader,
            val_loader,
            optimizer_cnn_lstm_attention_1,
            criterion_cnn_lstm_attention_1,
            num_epochs,
        )
    )
    # save the cnn_lstm_attention_1 model learning curve
    fig_cnn_lstm_attention_1_lr.savefig(
        os.path.join(figure_folder, "cnn_lstm_attention_model_learning_curve.png")
    )
    # test the cnn_lstm_attention_1 model
    (
        acc_cnn_lstm_attention_1,
        f1_cnn_lstm_attention_1,
        fig_cnn_lstm_attention_1,
        _,
        fig_attention_score,
        _,
    ) = mdl.evaluate_model_attn(
        cnn_lstm_attention_1_trained,
        test_loader,
    )
    # save the cnn_lstm_attention_1 model performance
    fig_cnn_lstm_attention_1.savefig(
        os.path.join(figure_folder, "cnn_lstm_attention_model_performance.png")
    )
    # save the attention score
    fig_attention_score.savefig(
        os.path.join(figure_folder, "cnn_lstm_attention_model_attention_score.png")
    )

    # # --------------------------------------------------
    # # CNN + LSTM + Attention model
    # cnn_lstm_attention = mdl.CNNLSTMAttn()
    # optimizer_cnn_lstm_attention = Adam(
    #     cnn_lstm_attention.parameters(), lr=learning_rate_cnnlstmatn, weight_decay=1e-3
    # )
    # sheduler_cnn_lstm_attention = StepLR(
    #     optimizer_cnn_lstm_attention, step_size=4, gamma=0.1
    # )  # learning rate decay
    # criterion_cnn_lstm_attention = CrossEntropyLoss()
    # cnn_lstm_attention_trained, fig_cnn_lstm_attention_lr, _ = mdl.train_model_lr_decay(
    #     cnn_lstm_attention,
    #     train_loader,
    #     val_loader,
    #     optimizer_cnn_lstm_attention,
    #     criterion_cnn_lstm_attention,
    #     sheduler_cnn_lstm_attention,
    #     num_epochs=15,
    # )
    # # save the cnn_lstm_attention model learning curve
    # fig_cnn_lstm_attention_lr.savefig(
    #     os.path.join(figure_folder, "cnn_lstm_attention_model_learning_curve.png")
    # )
    # # test the cnn_lstm_attention model
    # acc_cnn_lstm_attention, f1_cnn_lstm_attention, fig_cnn_lstm_attention, _ = (
    #     mdl.evaluate_model(
    #         cnn_lstm_attention_trained,
    #         test_loader,
    #     )
    # )
    # # save the cnn_lstm_attention model performance
    # fig_cnn_lstm_attention.savefig(
    #     os.path.join(figure_folder, "cnn_lstm_attention_model_performance.png")
    # )

    # Ablation study
    print("Ablation study")
    lstm = mdl.SimpleLSTM()
    cnn_lstm = mdl.CNNLSTM()
    optimizer_lstm = Adam(
        lstm.parameters(), lr=learning_rate_ablation, weight_decay=1e-3
    )
    optimizer_cnn_lstm = Adam(
        cnn_lstm.parameters(), lr=learning_rate_ablation, weight_decay=1e-3
    )
    criterion_ablation = CrossEntropyLoss()
    print("LSTM model training")
    lstm_trained, fig_lstm_lr, _, _ = mdl.train_model(
        lstm,
        train_loader,
        val_loader,
        optimizer_lstm,
        criterion_ablation,
        num_epochs,
    )
    # save the lstm model learning curve
    fig_lstm_lr.savefig(os.path.join(figure_folder, "lstm_model_learning_curve.png"))
    # test the lstm model
    acc_lstm, f1_lstm, fig_lstm, _ = mdl.evaluate_model(
        lstm_trained,
        test_loader,
    )
    # save the lstm model performance
    fig_lstm.savefig(os.path.join(figure_folder, "lstm_model_performance.png"))

    print("CNN + LSTM model training")
    cnn_lstm_trained, fig_cnn_lstm_lr, _, _ = mdl.train_model(
        cnn_lstm,
        train_loader,
        val_loader,
        optimizer_cnn_lstm,
        criterion_ablation,
        num_epochs,
    )
    # save the cnn_lstm model learning curve
    fig_cnn_lstm_lr.savefig(
        os.path.join(figure_folder, "cnn_lstm_model_learning_curve.png")
    )
    # test the cnn_lstm model
    acc_cnn_lstm, f1_cnn_lstm, fig_cnn_lstm, _ = mdl.evaluate_model(
        cnn_lstm_trained,
        test_loader,
    )
    # save the cnn_lstm model performance
    fig_cnn_lstm.savefig(os.path.join(figure_folder, "cnn_lstm_model_performance.png"))

    # baseline_train_acc = 0.0
    # acc_baseline = 0.0
    # mymodel_train_acc = 0.0
    # acc_cnn_lstm_attention_1 = 0.0

    # # print results
    # print("Experiment results")
    # print("Baseline model accuracy: ", acc_baseline)
    # print("Baseline model f1 score: ", f1_baseline)
    # print("CNN + LSTM + Attention model accuracy: ", acc_cnn_lstm_attention_1)
    # print("CNN + LSTM + Attention model f1 score: ", f1_cnn_lstm_attention_1)
    # print("Ablation study")
    # print("LSTM model accuracy: ", acc_lstm)
    # print("LSTM model f1 score: ", f1_lstm)
    # print("CNN + LSTM model accuracy: ", acc_cnn_lstm)
    # print("CNN + LSTM model f1 score: ", f1_cnn_lstm)

    return baseline_train_acc, mymodel_train_acc, acc_baseline, acc_cnn_lstm_attention_1


if __name__ == "__main__":
    baseline_acc_train, mymodel_acc_train, baseline_acc_test, mymodel_acc_test = (
        sentiment_analysis()
    )
    print("TA shows the training and test accuracy of the baseline model")
    print("TB shows the training and test accuracy of the CNN + LSTM + Attention model")
    print(
        "TA:{},{};TB:{},{};".format(
            baseline_acc_train,
            baseline_acc_test,
            mymodel_acc_train,
            mymodel_acc_test,
        )
    )
