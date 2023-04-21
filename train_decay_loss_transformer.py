import numpy as np
from grid_LSTM import netLSTM, netLSTM_full
from grid_data_v2 import TextDataset
import grid_data_v2 as grid_data
from grid_configuration import config
from util import Record, save_var, get_file_list, Regeneralize, list_to_csv
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import time
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from tst import Transformer
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

plt.ion()

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.cuda.set_device(config.deviceID)
print(config.test_ID)
PATH = config.path
if not os.path.exists(PATH):
    os.mkdir(PATH)
 
'''Parmaters used in Transformer.'''
ERROR_PER = config.ERROR_PER
NE = config.ne  # number of ensemble
GAMMA = config.GAMMA
T = config.T
d_model = 12 # Lattent dim
q = 8 # Query size
v = 8 # Value size
h = 2 # Number of heads
N = 1 # Number of encoder and decoder to stack
attention_size = None#12 # Attention window size
dropout = 0.2 # Dropout rate
pe = 'regular' # Positional encoding
chunk_mode = None

# d_input = 96  # From dataset
# d_output = 24  # From dataset
d_input = 9  # From dataset
d_output = 1  # From dataset

''' load data'''
text = TextDataset(config)
textLoader = DataLoader(text, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, drop_last=config.drop_last)
criterion = torch.nn.MSELoss()

def predict(data, params=None, model_predict=None): # predict 24h each time，and join the predictions to get the final result
    result = []
    input_ = torch.tensor(data)
    input_ = Variable(input_.view(1, len(data), config.input_dim).float()).cuda()
    # if params is not None:
    #     model_predict.set_parameter(params)
    i = 0
    while i <= len(data) - config.train_len:
        # pred = model_predict(input_[:, i:i+config.train_len, :].transpose(1,2))
        # result.append(pred[:, 0, :])
        pred = model_predict(input_[:, i:i+config.train_len, :])
        # result.append(pred[0])
        # pred = input_[:, i:i+config.train_len, :]
        result.append(pred[:, -24:, :])
        # print('predicting: {} to {}'.format(i, i + config.train_len))
        i += 24
    #save_var(result, 'result')
    return torch.cat(result, dim=1)

def evaluate(model, epoch): # evaluate the performance of the model in test set
    for i, k in enumerate(config.test_ID):
        input_ = text.test_data_input_list[i] # ->array(data_len * in_dim)
        target = text.test_data_output_list[i]# ->array(data_len * 1)
        raw_data = pd.read_csv("D:/gjx/cyt/TgDLF/grid_TgDLF1/{}.csv".format(config.data_list[k-1]))
        real_std = raw_data.LOAD.std()
        real_mean = raw_data.LOAD.mean()
        raw = np.array(raw_data.LOAD)[config.predict_len:-config.predict_len]
        pred = predict(input_, params=None, model_predict=model)
        # add the average trend to get the real ratio
        train_len = config.train_len - 24
        if grid_data.use_mean_ratio == True:
            if grid_data.use_different_mean_ratio == True:
                mean_ratio = text.mean_ratio_all_pred[k-1,:]
            elif grid_data.use_CV_ratio == True:
                mean_ratio = text.mean_ratio_group_pred[config.test_set_ID-1,:]                
            else:
                mean_ratio = text.mean_ratio_all_ave_pred
            print('mean_ratio:', mean_ratio)
            pred = pred.cpu().detach().numpy()[0] + mean_ratio.reshape(-1, 1)[train_len:] 
            print('test_pred:', np.shape(pred))
            target = target[train_len:] + mean_ratio.reshape(-1,1)[train_len:]
            loss = criterion(torch.tensor(pred[:, 0]).float(), torch.tensor(target[:, 0]).float())
        else:
            print(pred.shape)
            pred = pred.cpu().detach().numpy()[0]
            target = target[train_len:]
            loss = criterion(torch.tensor(pred).float(), torch.tensor(target[:, 0]).float())
        print("ID{}\t test loss: {}".format(k, loss))
        mean = grid_data.load_mean[k - 1][0]
        std = grid_data.load_std[k - 1][0]
        pred_ratio = Regeneralize(np.array(pred[:, 0]), mean, std)
        pred_real = pred_ratio * raw[train_len:]
        target_ratio = Regeneralize(target, mean, std).reshape(-1,1)
        target_real = target_ratio * raw[train_len:].reshape(-1,1)
        loss_ratio = criterion(torch.tensor(pred_ratio).float(), torch.tensor(target_ratio[:, 0]).float()) #new
        print("ID{}\t ratio loss: {}".format(k, loss_ratio))
        # make a normalization of real load value:
        loss_relative = np.mean(np.abs(pred_real - target_real.reshape(-1))/target_real.reshape(-1))
        std = 1 * pred_real.std(0)
        pred_normalized = (pred_real - real_mean) / real_std
        print('pred_normalized:', pred_normalized)
        target_normalized = (target_real.reshape(-1) - real_mean) / real_std
        print('pred_normalized shape:', np.shape(pred_normalized))
        print('target_normalized shape:', np.shape(target_normalized))
        loss_real = criterion(Variable(torch.tensor(pred_normalized).float()),
                              Variable(torch.tensor(target_normalized).float()))
        print("ID{}\t relative loss: {}".format(k, loss_relative))
        print("ID{}\t real loss: {}".format(k, loss_real))
        f =  open(r'{}/epoch{}_test_loss_{}.csv'.format(PATH, epoch, config.experiment_ID), 'a')
        f.write('{},{},{},{},{},{}\n'.format(k, loss, loss_ratio, loss_real, loss_relative, std.mean()))
        f.close()
    


def run(): # train and evaluate the model
    with open('{}/time.txt'.format(PATH), 'a') as f:
        f.write('{},\n'.format(time.strftime('%Y%m%d_%H_%M_%S')))
    f.close()
    model = Transformer(d_input, d_model, d_output,
                        q, v, h, N, attention_size=attention_size,
                        dropout=dropout, chunk_mode=chunk_mode, pe=pe)

    model = model.cuda()
    criterion = torch.nn.MSELoss()
    # decay the learning rate
    for epoch in range(config.epoch):
        if epoch<=50:
            lr_tmp = 1e-2
        elif epoch>50 and epoch<=100:
            lr_tmp = 5e-3
        elif  epoch>100 and epoch<=150:
            lr_tmp = 3e-3
        elif epoch > 150 and epoch <= 200:
            lr_tmp = 3e-3
        elif epoch > 200 and epoch <= 250:
            lr_tmp = 1e-3
        else:
            lr_tmp = 3e-4
        opt = torch.optim.Adam(model.parameters(), lr=lr_tmp)
        # training
        for i, data in enumerate(textLoader):
            print('#'*30)
            print("{}: batch{}".format(time.strftime('%Y%m%d_%H_%M_%S'), i))
            input_, target = data
            # if i == 10:
            #     print(data)
            input_, target = map(Variable, (input_.float(), target.float()))
            target = target[:, -config.predict_len:, :]
            print(target.shape)
            target = target.reshape(-1, config.output_dim)
            # input_ = input_.transpose(1, 2)
            input_ = input_.cuda()
            target = target.cuda()
            pred = model(input_)
            pred = pred[:,-config.predict_len:, :].reshape(-1, config.output_dim)
            loss = criterion(pred, target)
            print(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            with open(PATH+'/time.txt', 'a') as f:
                f.write(time.strftime('%Y%m%d_%H_%M_%S') + ',' + str(loss) + ',\n')
            f.close()            
    # save model
    torch.save(model,'E6101/t1.pt')
    # evaluate the trained model
    evaluate(model, 4)



if __name__ == '__main__':
    start = time.time()
    run() # the training process based on the Transformer model.
    # model = torch.load('E6101/trans.pt')
    # evaluate(model, 4) # only evaluate
    print(config.test_ID)
    print(time.time() - start)

