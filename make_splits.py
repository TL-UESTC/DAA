import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

class AliExpressDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.categorical_data = data[:, :16].astype(np.int)
        self.numerical_data = data[:, 16: -2].astype(np.float32)
        self.labels = data[:, -2:].astype(np.float32)
        self.numerical_num = self.numerical_data.shape[1]
        self.field_dims = np.max(self.categorical_data, axis=0) + 1

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return self.categorical_data[index], self.numerical_data[index], self.labels[index]

def main(dataset_name,
         dataset_path,
         batch_size,
         chunksize,
         version,
         rate):

    all_pos_clk = 0
    all_neg = 0
    all_pos_action = 0
    all_pos_clk_te = []
    all_neg_te = []
    all_pos_action_te = []
    print(dataset_name)

    for epoch_i in range(1):
        train_datas = pd.read_csv(os.path.join(dataset_path, dataset_name) + '/train.csv', chunksize=batch_size * chunksize)
        for i, data in enumerate(train_datas):
            data = data.to_numpy()[:, 1:]
            train_dataset = AliExpressDataset(data)
            train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

            for i, (_, _, labels) in enumerate(train_data_loader):
                clk=(labels[:,0] == 1)
                all_pos_clk += clk.sum()
                all_pos_clk_te.extend(clk.numpy())
                action = (labels[:,1] == 1)
                all_pos_action += action.sum()
                all_pos_action_te.extend(action.numpy())
                neg = (labels[:,0] == 0)
                all_neg += neg.sum()
                all_neg_te.extend(neg.numpy())
    
    print(all_pos_clk)
    print(all_neg)
    print(all_pos_action)

    ###################
    print(len(all_pos_clk_te), sum(all_pos_clk_te))
    print(len(all_neg_te), sum(all_neg_te))
    print(len(all_pos_action_te), sum(all_pos_action_te))

    all_pos_clk_te = np.array(all_pos_clk_te)
    all_neg_te = np.array(all_neg_te)
    all_pos_action_te = np.array(all_pos_action_te)
    print("Before random drop:")
    print(all_pos_clk_te.sum())
    index = (np.argwhere(all_pos_clk_te == True)).reshape([-1])
    droplistlen = (np.random.random_sample((all_pos_clk_te.sum(),)) > rate).sum()
    drop = np.random.choice(index, droplistlen,replace=False)
    all_pos_clk_te[drop] = False
    
    print(all_neg_te.sum())
    index = (np.argwhere(all_neg_te == True)).reshape([-1])
    droplistlen = (np.random.random_sample((all_neg_te.sum(),)) > rate).sum()
    drop = np.random.choice(index, droplistlen,replace=False)
    all_neg_te[drop] = False

    print("After random drop:")
    print(all_pos_clk_te.sum())
    print(all_neg_te.sum())

    all_chosen = all_pos_clk_te|all_neg_te

    action = all_chosen&all_pos_action_te
    print(action.sum())

    start=0
    if os.path.isfile(os.path.join(dataset_path, dataset_name) + '/split_'+version+'_train.csv'):
        os.remove(os.path.join(dataset_path, dataset_name) + '/split_'+version+'_train.csv')

    for epoch_i in range(1):
        train_datas = pd.read_csv(os.path.join(dataset_path, dataset_name) + '/train.csv', chunksize=batch_size * chunksize)
        # test_datas = pd.read_csv(os.path.join(dataset_path, dataset_name) + '/test.csv', chunksize=batch_size * chunksize)
        for i, data_ori in enumerate(train_datas):
            tmp_mask = all_chosen[start:start+data_ori.shape[0]]
            masked_data = data_ori[tmp_mask]
            if start==0:
                masked_data.to_csv(os.path.join(dataset_path, dataset_name) + '/split_'+version+'_train.csv',index=False)
            else:
                masked_data.to_csv(os.path.join(dataset_path, dataset_name) + '/split_'+version+'_train.csv',mode='a', header=False,index=False)
            start+=data_ori.shape[0]

    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='AliExpress_NL',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US','AliExpress_RU'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--chunksize', type=int, default=2048)
    parser.add_argument('--version', default='v2')
    parser.add_argument('--rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.batch_size,
         args.chunksize,
         args.version,
         args.rate)