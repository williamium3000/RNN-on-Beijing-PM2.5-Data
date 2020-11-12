import data_preprocessing
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import copy
import os
import time
import json
import logging


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        dimension_of_one_input = 8
        num_of_layers = 2
        dropout = 0.5# 0 for no dropout
        hidden_size = 100
        # self.rnn = nn.LSTM(input_size = dimension_of_one_input, hidden_size = hidden_size, num_layers = num_of_layers, batch_first = True, dropout = dropout)
        # self.rnn = nn.RNN(input_size = dimension_of_one_input, hidden_size = hidden_size, num_layers = num_of_layers, batch_first = True, dropout = dropout)    
        self.rnn = nn.GRU(input_size = dimension_of_one_input, hidden_size = hidden_size, num_layers = num_of_layers,batch_first = True, dropout = dropout)   
        self.out = nn.Linear(hidden_size, 1)
        # nn.init.kaiming_uniform(self.out.weight)
    def forward(self, x):
        '''
        x.shape:(batch_num, time_step, dimension_of_one_input)
        '''
        r_out, (h_n, h_c) = self.rnn(x, None)
        # r_out, h_n = self.rnn(x, None)
        out = self.out(r_out[:, -1, :]).squeeze()
        return out

np.random.seed(0)
torch.manual_seed(0)# 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(0)# 为所有的GPU设置种子，以使得结果是确定的

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def check_accuracy(device, loader, model, phase):
    loss_func = nn.L1Loss("mean")
    logging.info('Checking loss on %s set: ' % phase)
    model.eval()
    num_samples = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            loss = loss_func(scores, y)
            total_loss += loss.item() * x.size(0)
            num_samples += x.size(0)
        total_loss = total_loss / num_samples
        logging.info("{} loss: {}".format(phase, total_loss))
        return total_loss


def train(model, optimizer, dataloaders, device, epochs):
    loss_func = nn.L1Loss("mean")
    rec = []
    model = model.to(device=device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10e10
    for e in range(epochs):
        for t, (x, y) in enumerate(dataloaders['train']):
            model.train()
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            loss = loss_func(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info('epoche %d, loss = %f' % (e, loss.item()))
        train_loss = check_accuracy(device, dataloaders['train'],
            model, 'train')
        test_loss = check_accuracy(device, dataloaders['val'], model,
            'validate')
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        rec.append((loss.item(), train_loss, test_loss))
    logging.info('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    save_model(save_dir='model_checkpoint', whole_model=False, file_name=task_name,
        model=model)
    return model, best_loss, rec


def save_model(save_dir, whole_model, file_name=None, model=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if file_name:
        save_path = os.path.join(save_dir, file_name)
    else:
        save_path = os.path.join(save_dir, str(int(time.time())))
    if model:
        if whole_model:
            torch.save(model, save_path + '.pkl')
        else:
            torch.save(model.state_dict(), save_path + '.pkl')
    else:
        logging.info('check point not saved, best_model is None')

task_name = "PRSA_GRU"
optimizer_name = 'Adam'
lr = 0.0001
batch_size = 256
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
epochs = 50

logging.basicConfig(filename="{}.log".format(task_name), level=logging.INFO)

logging.info(
    """{}:
    - optimizer: {}
    - learning rate: {}
    - batch size: {}
    - device : {}
    - epochs: {}
""".format(task_name,  
            optimizer_name, 
            lr, 
            batch_size,
            device,  
            epochs)
)

if __name__ == "__main__":
    rnn = Model()
    params_to_update = []
    for name, param in rnn.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    optimizer = getattr(optim, optimizer_name)(params_to_update, lr=lr)
    dataLoaders = {
        "train": data_preprocessing.train_data_loader, 
        "val": data_preprocessing.test_data_loader
                }
                    

    best_model, best_loss, rec = train(model=rnn, optimizer=optimizer, dataloaders=dataLoaders, device=device,
        epochs=epochs)
    json.dump(rec, open('{}.json'.format(task_name), 'w'))






    
    

