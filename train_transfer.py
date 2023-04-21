import numpy as np
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
from scipy import signal
plt.ion()

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
    
torch.cuda.set_device(config.deviceID)
PATH = config.path

# Model parameters
d_model = 12 # Lattent dim
q = 8 # Query size
v = 8 # Value size
h = 2 # Number of heads
N = 1 # Number of encoder and decoder to stack
attention_size = None#12 # Attention window size
dropout = 0.2 # Dropout rate
pe = 'regular' # Positional encoding
chunk_mode = None

d_input = 9  # From dataset
d_output = 1  # From dataset

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")


def ave_ratio (data_origin, use_filter=True, test_len=0): # extract the dimensionless trend from the source data 
    mean_ratio_all = None
    load = data_origin
    load_raw_array = load[:, 0]
    input_load = np.array(load_raw_array)
    data_num = np.shape(input_load)[0]
    if test_len != 0:
        week_num_all = int(data_num/168)
        data_num_all = data_num
        input_load = input_load[:-test_len]
        data_num = np.shape(input_load)[0]
    week_num = int(data_num/168) # calculate the number of weeks
    delet_ID = np.arange(week_num*168, data_num)

    input_load_del = np.delete( input_load, delet_ID, 0)
    input_load_week = input_load_del.reshape(week_num,168)
    # calculate the average ratio in one week
    input_load_week_mean = np.mean(input_load_week, axis=0)
    print('original:',np.mean(input_load_week_mean))
    if use_filter == True:
        b, a = signal.butter(8, 0.2, 'lowpass')
        filter_input_load_week_mean = signal.filtfilt(b, a, input_load_week_mean)
        filter_input_load_week_mean = (filter_input_load_week_mean-np.min(filter_input_load_week_mean)) / (np.max(filter_input_load_week_mean)-np.min(filter_input_load_week_mean))
        input_load_week_mean = filter_input_load_week_mean * (np.max(input_load_week_mean)-np.min(input_load_week_mean)) + np.min(input_load_week_mean)
        print('filtered:',np.mean(input_load_week_mean))
    # generate the average ratio for the length of data_num
        if test_len != 0:
            week_num = week_num_all
            data_num = data_num_all
        mean_ratio = None
        for i in range (week_num+1):
            if mean_ratio is None:
                mean_ratio = input_load_week_mean
            else:
                mean_ratio = np.hstack((mean_ratio, input_load_week_mean))
        delet_ID = np.arange(data_num, np.shape(mean_ratio)[0])
        mean_ratio = np.delete( mean_ratio, delet_ID, 0).reshape(1,-1)
        if mean_ratio_all is None:
            mean_ratio_all = mean_ratio
        else:
            mean_ratio_all = np.vstack((mean_ratio_all, mean_ratio))
    return load_raw_array,mean_ratio_all

def process_data(text,train_len=4704,DT=False,transfer=False,m_ex=None): # get the local fluctutaion for training Transformer
    tmp = text.train_data_input_all_list[0][:train_len]
    tmp_out = text.train_data_output_all_list[0][:train_len]
    m = None
    m_out = None
    if DT:
        if m_ex:
            m = m_ex[0]
            tmp[:,0] = tmp[:,0]-m[0]
            m_out = m_ex[1]
            tmp_out = tmp_out-m_out
        else:
            if transfer:
                l,m = ave_ratio(tmp)
                m = m[0]
                tmp[:,0] = tmp[:,0]-m
                m_out = np.append(m[24:],m[:24])
                tmp_out = tmp_out-m_out
            else:
                l,m = ave_ratio(tmp, test_len=1536)
                m = m[0]
                tmp[:,0] = tmp[:,0]-m
                m_out = np.append(m[24:],m[:24])
                tmp_out = tmp_out-m_out
    train_data_input = None
    train_data_output = None
    len_num = int((tmp.shape[0]/24)-(config.train_len/24))
    print('len_num:')
    print(len_num)
    train_data_input_temp = np.zeros((len_num, config.train_len, config.input_dim))
    for i in range (len_num):
        for j in range (config.input_dim):
            train_data_input_temp[i,:,j] = tmp[:,j][(i*24):((i*24)+config.train_len)]
    if train_data_input is None:
        train_data_input = train_data_input_temp
    else:
        train_data_input = np.concatenate((train_data_input, train_data_input_temp), axis = 0)
    train_data_output_temp = np.zeros((len_num, config.train_len, config.output_dim)) 
    for i in range (len_num):
        for j in range (config.output_dim):
            train_data_output_temp[i,:,j] = tmp_out[:,j][(i*24):((i*24)+config.train_len)]
    if train_data_output is None:
        train_data_output = train_data_output_temp
    else:
        train_data_output = np.concatenate((train_data_output, train_data_output_temp), axis = 0)
    text.train_data_input = train_data_input
    text.train_data_output = train_data_output
    text.line_num = train_data_input.shape[0]
    textLoader = DataLoader(text, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, drop_last=config.drop_last)
    return textLoader, m, m_out

def run(sourceLoader,textLoader,weather_error_train=0.05,weather_error_test=0.05,transfer=False, mode=0): # training process
    model = Transformer(d_input, d_model, d_output, 
                        q, v, h, N, attention_size=attention_size, 
                        dropout=dropout, chunk_mode=chunk_mode, pe=pe)
    model = model.cuda()
    criterion = torch.nn.MSELoss()
    lr_tmp = 1e-2
    opt = torch.optim.Adam(model.parameters(), lr=lr_tmp)
    source_result=[]
    target_result=[]
    loss_curve = []
    # training of source data
    if transfer:
        for epoch in range(config.epoch):
    #     for epoch in range(0):
            for i, data in enumerate(sourceLoader):
                input_, target = data
                input_[:, :, 1:5] = input_[:, :, 1:5] * (1 + np.random.randn() * weather_error_train)
#                 input_[:, :, 1:5] = 0
                input_, target = map(Variable, (input_.float(), target.float()))
                target = target[:, -config.predict_len:, :]
                target = target.reshape(-1, config.output_dim)
                input_ = input_.cuda()
                target = target.cuda()
                pred = model(input_)
                pred = pred[:,-config.predict_len:, :].reshape(-1, config.output_dim)
                loss = criterion(pred, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
        source_result.append(round(loss.item(),5))
        config.epoch = int(config.epoch/2)
        lr_tmp = 1e-3
        # three transfer strategies
        if mode == 0:
            opt = torch.optim.Adam(model.parameters(), lr=lr_tmp)
        elif mode == 1:
            for name,paras in model.named_parameters():
                if "layers" in name or "_embedding" in name:
                    paras.requires_grad = False
            opt = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=lr_tmp)
        elif mode == 2:
            for name,paras in model.named_parameters():
                if "_linear" in name:
                    paras.requires_grad = False
            opt = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=lr_tmp)
    # training of target data
    for epoch in range(config.epoch):
        for i, data in enumerate(textLoader):
#             if i==0:
#                 continue
            if i<=1:
                input_, target = data
                input_[:, :, 1:5] = input_[:, :, 1:5] * (1 + np.random.randn() * weather_error_train)
#                 input_[:, :, 1:5] = 0
                input_, target = map(Variable, (input_.float(), target.float()))
                target = target[:, -config.predict_len:, :]
#                 input_ = input_[:32]
#                 target = target[:32]
                target = target.reshape(-1, config.output_dim)
#                 input_=input_.transpose(1,2)
                input_ = input_.cuda()
                target = target.cuda()
                pred = model(input_)
                pred = pred[:,-config.predict_len:, :].reshape(-1, config.output_dim)
                loss = criterion(pred, target)
#                 print("train loss:",loss)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if len(target_result)==0:
                    target_result.append(round(loss.item(),5))
                elif epoch==config.epoch-1 and i==1:
                    target_result.append(round(loss.item(),5))                
            else:
                input_, target = data
                input_[:, :, 1:5] = input_[:, :, 1:5] * (1 + np.random.randn() * weather_error_test)
#                 input_[:, :, 1:5] = 0
                input_, target = map(Variable, (input_.float(), target.float()))
                target = target[:, -config.predict_len:, :]
#                 print(target.shape)
                target = target.reshape(-1, config.output_dim)
#                 input_=input_.transpose(1,2)
                input_ = input_.cuda()
                target = target.cuda()
                pred = model(input_)
                pred = pred[:,-config.predict_len:, :].reshape(-1, config.output_dim)
                loss = criterion(pred, target)
#                 print("test loss:",loss)
                if epoch==config.epoch-1:
                    target_result.append(round(loss.item(),5))
                if epoch % 10 ==0:
                    loss_curve.append(round(loss.item(),5))
    return source_result,target_result,loss_curve

def exper(transfer=False, dt=False, mode=0, weather_error_test=0.05): # 10 independent experiments
    seeds = [i for i in range(10)]
    nums = 10
    weather_error_train = 0.05
    source_result = []
    target_result = []
    loss_result = []
    for i in range(nums):
        seed = seeds[i]
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_device(config.deviceID)
        # get the source data
        config.train_ID=[8]
        source = TextDataset(config)
        sourceLoader = DataLoader(source, batch_size=config.batch_size, shuffle=False,
                                num_workers=config.num_workers, drop_last=config.drop_last)
        sourceLoader,m2,m2_out = process_data(source,DT=dt,transfer=transfer)
        # get the target data
        config.train_ID=[9]
        config.batch_size=64
        config.epoch=100
        ''' load data and initialize enn net'''
        text = TextDataset(config)
        if not transfer:
            textLoader,m1,m1_out = process_data(text,DT=dt,transfer=transfer)
        else:
            textLoader,m1,m1_out = process_data(text,DT=dt,transfer=transfer,m_ex=(m2,m2_out))
        s,t,l=run(sourceLoader,textLoader,weather_error_train,weather_error_test,transfer,mode)
        if len(s)!=0: source_result.append(s)
        if len(t)!=0: target_result.append(t)
        if len(l)!=0: loss_result.append(l)
    return source_result,target_result,loss_result

def final_run(dt = False): # experiments under diffenrent three transfer strategies and different weather noise rate
    poss_trans = [False,True]
    poss_mode = [0,1,2]
    poss_we = [0.05,0.1,0.2,0.3]
    result = {}
    for trans in poss_trans:
#         print(trans)
        if trans == False:
            for we in poss_we:
                s,t,l=exper(transfer=trans,dt=dt,mode=0,weather_error_test=we)
                key = (trans,0,we)
                r = [s,t,l]
                result[key]=r
        else:
            for mode in poss_mode:
                for we in poss_we:
                    s,t,l=exper(transfer=trans,dt=dt,mode=mode,weather_error_test=we)
                    key = (trans,mode,we)
                    r = [s,t,l]
                    result[key]=r
    return result

if __name__ == '__main__':
    result_DT = final_run(dt = True)
    f = open("new_result/fs_mtg_transdt_128.txt",'w')
    f.write(str(result_DT))
    f.close()