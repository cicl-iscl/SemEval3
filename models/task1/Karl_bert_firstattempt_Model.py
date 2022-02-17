# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:37:13 2021

@author: karlm
"""

import pandas as pd
import numpy as np
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



class BERT_Arch(nn.Module):

    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,512)
      
      #optinal layer
      self.fc3 = nn.Linear(512,512)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(512,2)

      #softmax activation function
      self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model  
      hidden = self.bert(sent_id, attention_mask=mask)
      
      #print(hidden[1])
      x = self.fc1(hidden[1])

      x = self.relu(x)

      x = self.dropout(x)
      
      x = self.fc3(x)
      
      x = self.relu(x)
      
      
      x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      return x
  
    

  
 
        
  