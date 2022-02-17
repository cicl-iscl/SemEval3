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
      self.dropout = nn.Dropout(0.2)
      
      # relu activation function
      self.relu =  nn.Tanh()

      # dense layer 1
      self.fc1 = nn.Linear(768,16)
      
      #optinal
      self.fc3 = nn.Linear(512,512)
      
      #optinal
      self.fc4 = nn.Linear(512,512)
      
      #optinal
      self.fc5 = nn.Linear(2048,2048)
      
      #optinal
      self.fc6 = nn.Linear(2048,2048)
      
      # dense layer 2 (Output layer)
      self.fc2 = nn.Linear(16,1, bias = True)

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
      
      #x = self.fc3(x)
      
      #x = self.relu(x)
      
      #x = self.dropout(x)
      
      #x = self.fc4(x)
      
      #x = self.relu(x)
      
      #x = self.dropout(x)
      
      #x = self.fc5(x)
      
      #x = self.relu(x)
      
      #x = self.dropout(x)
      
      #x = self.fc6(x)
      
      #x = self.relu(x)
      
      #x = self.dropout(x)

      # output layer
      x = self.fc2(x)
      
     
      
      
      return x
  
    

  
 
        
  