import torch
import numpy as np
import random


class DefaultConfiguration: # configuration of training
    def __init__(self):
        # experimental parameter settings
        self.seed = 666
        self.supplement = 'test_'
        self.experiment_ID = '6101'
        self.test_set_ID = 1
        self.deviceID = 0
        #self.deviceID = self.test_set_ID % 2
        self.epoch = 300
        self.ERROR_PER = 0.02
        # self.path = 'E' + self.experiment_ID
        self.path = 'E' + self.experiment_ID
        self.GAMMA = 10
        self.drop_last = False
        
        # dataset parameters
        self.test_pro = 0.3
        self.total_well_num = 14
        self.train_len = int(24*4)  # the length of training data is 96, 4 days
        self.predict_len = int(24 * 1)
        
        # network parameters, it can also be reset in train_decay_loss_transformer.py
        self.T = 1
        self.ne = 100
        self.use_annual = 1
        self.use_quarterly = 0
        self.use_monthly = 0      
        self.input_dim = 9  
        self.hid_dim = 30  
        self.num_layer = 1  
        self.drop_out = 0.3
        self.output_dim = 1
        
        # training parameters
        self.batch_size = 512
        self.num_workers = 0
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        #self.max_epoch = 30
        self.display_step = 10
        self.reverse = False   
       
        # cross validation experiment
        self.test_well_num = int(self.test_pro*self.total_well_num)
        self.train_well_num = self.total_well_num - self.test_well_num
        self.rdn_lst = np.arange(1, self.total_well_num+1, 1)
        random.shuffle(self.rdn_lst)
        a = [[5, 4, 1], [6, 8, 3], [12, 9, 7], [13, 2, 10]]
        b = [[6, 8, 3, 12, 9, 7, 13, 2, 10],
             [5, 4, 1, 12, 9, 7, 13, 2, 10],
             [5, 4, 1, 6, 8, 3,  13, 2, 10],
             [5, 4, 1, 6, 8, 3, 12, 9, 7]]
        self.test_ID = a[self.test_set_ID]
        self.train_ID = b[self.test_set_ID]
        #self.test_ID = np.array(self.rdn_lst[0:self.test_well_num])
        #self.train_ID = np.array(self.rdn_lst[self.test_well_num:self.total_well_num])
        self.data_list = ['CY', 'HD', 'FT', 'SJS', 'PG', 'YZ', 'CP',
                          'MTG', 'FS', 'DX', 'HR', 'MY', 'SY', 'YQ']
        #self.test_ID = [9, 5, 12, 2]

                
        # training infos

        self.info = "e{}-{}-{}-{}-{}-{}".format(self.experiment_ID, self.path, str(self.hid_dim), str(self.batch_size),
                                                str(self.ne), str(self.ERROR_PER))
        
        if torch.cuda.is_available():
            self.use_gpu = True
        else:
            self.use_gpu = False  
            
        # the setting of load and save model, and predict
        self.print_freq = 10
        self.checkpoint = 'grid_'+self.supplement+self.experiment_ID+'.pkl'
        
    def update(self):
        
        random.shuffle(self.rdn_lst)
        self.experiment_ID = str(int(self.experiment_ID) + 1)
        self.checkpoint = 'well'+self.supplement+'{:d}.pkl'.format(int(self.experiment_ID))        


config = DefaultConfiguration()
