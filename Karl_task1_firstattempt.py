# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 10:25:50 2021

@author: karlm
"""


import os, sys, time
import pandas as pd
import numpy as np
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from transformers import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



from Karl_bert_firstattempt_Model import BERT_Arch

# specify GPU
device = torch.device("cuda")

#This code is based on the tutorial at: https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/

#loading the data
basedata = pd.read_csv("./data/train/train_subtask-1/en/En-Subtask1-fold_0.tsv", sep ='\t', )
print(basedata.head())
testdata = pd.read_csv("./data/train/train_subtask-1/en/En-Subtask1-fold_1.tsv", sep ='\t', )
print(testdata.head())


#loading the pretrained bert model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# freeze parameters of pretrained network only output layers will be trained
#for param in bert.parameters():
#    param.requires_grad = False


#running the tokenizer
tokens_train = tokenizer.batch_encode_plus(
    basedata.Sentence.tolist(),
    max_length = 15,
    padding = 'max_length',
    truncation=True
)

tokens_test = tokenizer.batch_encode_plus(
    testdata.Sentence.tolist(),
    max_length = 15,
    padding = 'max_length',
    truncation=True
)


#creating the tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(basedata.Labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(testdata.Labels.tolist())

#define a batch size
batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

#creating the model and pushing it to gpu
model = BERT_Arch(bert)
model = model.to(device)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-5)         



# define the loss function
cross_entropy  = nn.CrossEntropyLoss() 

# number of training epochs
epochs = 10

# function to train the model
def train():
  
  model.train()

  total_loss, total_accuracy = 0, 0
  
  # empty list to save model predictions
  total_preds=[]
  
  # iterate over batches
  for step,batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]
 
    sent_id, mask, labels = batch

    # clear previously calculated gradients 
    model.zero_grad()        

    # get model predictions for the current batch
    #print(mask)
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    #print(labels)
    #print(preds.size())
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
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)
  
  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds





# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(epochs):
     
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    #evaluate model
    #valid_loss, _ = evaluate()
    
    #save the best model
    #if valid_loss < best_valid_loss:
    #    best_valid_loss = valid_loss
    #    torch.save(model.state_dict(), 'saved_weights.pt')
    
    # append training and validation loss
    train_losses.append(train_loss)
    #valid_losses.append(valid_loss)
    
    print(f'\nTraining Loss: {train_loss:.3f}')
    #print(f'Validation Loss: {valid_loss:.3f}')
    

with torch.no_grad():
  preds_test = model(test_seq.to(device), test_mask.to(device))
  preds_test = preds_test.detach().cpu().numpy()
  
preds_test = np.argmax(preds_test, axis = 1)
print(classification_report(test_y, preds_test))




