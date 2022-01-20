# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:25:50 2021

@author: karl vetter
"""
import csv

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AdamW, CamembertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from Karl_bert_firstattempt_Model import BERT_Arch

# specify GPU
device = torch.device("cuda")

EPOCHS = 100
LEARNINGRATE = 1e-5
PATIENCE = 5

MAXLENGTH_EN = 15
MAXLENGTH_FR = 20
MAXLENGTH_IT = 15

# define the loss function
cross_entropy = nn.CrossEntropyLoss()

# number of training epochs
epochs = EPOCHS


# function to train the model
def train(train_dataloader, model, optimizer):
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        # progress update after every 50 batches.
        # if step % 50 == 0 and not step == 0:
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]

        sent_id, mask, labels = batch

        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        # print(mask)
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        # print(labels)
        # print(preds.size())
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds = preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds


def validate(test_dataloader, model):
    model.eval()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            batch = [r.to(device) for r in batch]
            sent_id, mask, labels = batch
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)
        avg_loss = total_loss / len(train_dataloader)
        total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds


for lang in ("fr", "it", "en"):
    outputstorage = np.zeros((3, 5))

    torch.cuda.empty_cache()

    data = pd.concat([pd.read_csv(f"data/train/train_subtask-1/{lang}/{lang.capitalize()}-Subtask1-fold_{i}.tsv",
                                  sep="\t") for i in range(3)])
    basedata, testdata = train_test_split(data, test_size=0.3)
    print(basedata.head())
    print(np.shape(basedata))
    print(testdata.head())
    print(np.shape(testdata))

    # loading the pretrained bert model and tokenizer
    if lang == "en":
        bert = AutoModel.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        maxlength = MAXLENGTH_EN
    elif lang == "fr":
        bert = AutoModel.from_pretrained('camembert-base')
        tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        maxlength = MAXLENGTH_FR
    elif lang == "it":
        bert = AutoModel.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0')
        tokenizer = BertTokenizerFast.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0')
        maxlength = MAXLENGTH_IT


    # running the tokenizer
    tokens_train = tokenizer.batch_encode_plus(
        basedata.Sentence.tolist(),
        max_length=maxlength,
        padding='max_length',
        truncation=True
    )

    tokens_test = tokenizer.batch_encode_plus(
        testdata.Sentence.tolist(),
        max_length=maxlength,
        padding='max_length',
        truncation=True
    )

    # creating the tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(basedata.Labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(testdata.Labels.tolist())

    # define a batch size
    batch_size = 32

    train_data = TensorDataset(train_seq, train_mask, train_y)
    # sampler for sampling the data during training
    train_sampler = RandomSampler(train_data)

    # dataLoader for train set
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_seq, test_mask, test_y)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # creating the model and pushing it to gpu
    model = BERT_Arch(bert)
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LEARNINGRATE)

    # set initial loss to infinite
    best_valid_loss = float('inf')

    # empty lists to store training and validation loss of each epoch
    train_losses = []
    valid_losses = []

    # for each epoch
    counter = 0
    for epoch in range(epochs):
        print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

        # train model
        train_loss, _ = train(train_dataloader, model, optimizer)

        # evaluate model
        valid_loss, _ = validate(test_dataloader, model)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1

        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')

        # Early stopping
        if counter >= PATIENCE:
            break

    data_test = pd.read_csv(f"data/test/official_test_sets/subtask-1/{lang.capitalize()}-Subtask1-test.tsv", sep="\t")
    # running the tokenizer
    tokens_data_test = tokenizer.batch_encode_plus(
        data_test.Sentence.tolist(),
        max_length=maxlength,
        padding='max_length',
        truncation=True
    )

    data_test_seq = torch.tensor(tokens_data_test['input_ids'])
    data_test_mask = torch.tensor(tokens_data_test['attention_mask'])

    print(tokens_data_test, data_test_seq, data_test_mask)


    with torch.no_grad():
        torch.cuda.empty_cache()
        model.eval()
        preds_test = model(data_test_seq.to(device), data_test_mask.to(device))
        preds_test = preds_test.detach().cpu().numpy()
        # preds_test = model(data_test_seq, data_test_mask)
        # preds_test = preds_test.numpy()

    preds_test = np.argmax(preds_test, axis=1)

    with open('answer.tsv', 'a') as f:
        f.write("ID\tLabels")
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(zip(data_test.ID.tolist(), preds_test))

    """accuracy = accuracy_score(test_y, preds_test)
    precision = precision_score(test_y, preds_test)
    recall = recall_score(test_y, preds_test)
    f1 = f1_score(test_y, preds_test, average='binary')
    f1macro = f1_score(test_y, preds_test, average='macro')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 (micro): {f1}")
    print(f"F1 (macro): {f1macro}")"""
