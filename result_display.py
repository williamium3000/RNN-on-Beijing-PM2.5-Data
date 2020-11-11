import data_preprocessing
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import data_preprocessing

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        dimension_of_one_input = 8
        num_of_layers = 100
        dropout = 0# 0 for no dropout
        hidden_size = 32
        # self.rnn = nn.LSTM(input_size = dimension_of_one_input, hidden_size = hidden_size, num_layers = num_of_layers, batch_first = True, dropout = dropout)
        self.rnn = nn.RNN(input_size = dimension_of_one_input, hidden_size = hidden_size, num_layers = num_of_layers, batch_first = True, dropout = dropout)    
        
        # nn.init.orthogonal(self.rnn.weight_ih_l0)  
        # nn.init.orthogonal(self.rnn.weight_hh_l0)   
        
        # self.rnn = nn.GRU(input_size = dimension_of_one_input, hidden_size = hidden_size, num_layers = num_of_layers,batch_first = True, dropout = dropout)   
        self.out = nn.Linear(hidden_size, 1)
        # nn.init.kaiming_uniform(self.out.weight)
    def forward(self, x):
        '''
        x.shape:(batch_num, time_step, dimension_of_one_input)
        '''
        # r_out, (h_n, h_c) = self.rnn(x, None)
        r_out, h_n = self.rnn(x, None)
        # r_out, h_n = self.rnn(x, None)
        out = self.out(r_out[:, -1, :]).squeeze()
        return out

def load_model(path, whole_model, model = None):
    if whole_model:
        model = torch.load(path)
    else:
        model.load_state_dict(torch.load(path))
    return model


model = Model()
model = load_model(path = "model_checkpoint/best_model_checkpoint.pkl", whole_model = False, model = model)
x, y = next(data_preprocessing.test_data_loader)
scores = model(x)
print(scores)