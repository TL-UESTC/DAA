from dis import dis
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from datasets.aliexpress import AliExpressDataset as AliExpressDataset1
from models.sharedbottom import SharedBottomModel
from models.singletask import SingleTaskModel
from models.omoe import OMoEModel
from models.mmoe_da import MMoEModel
from models.ple import PLEModel
from models.aitm import AITMModel
from models.metaheac import MetaHeacModel
from models.discriminator import Discriminator

import torch.autograd as autograd
from torch.autograd import Variable
import warnings

warnings.filterwarnings('ignore')
import time


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


def get_dataset(name, path):
    if 'AliExpress' in name:
        return AliExpressDataset1(path)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, categorical_field_dims, numerical_num, task_num, expert_num, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    if name == 'sharedbottom':
        print("Model: Shared-Bottom")
        return SharedBottomModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                                 tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'singletask':
        print("Model: SingleTask")
        return SingleTaskModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                               tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'omoe':
        print("Model: OMoE")
        return OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        return MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        return PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                        tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2),
                        specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        return AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                         tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'metaheac':
        print("Model: MetaHeac")
        return MetaHeacModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256),
                             tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, critic_num=5,
                             dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


class EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model.state_dict(), self.save_path)
            print('-' * 20, 'Save Model Success', '-' * 20)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def calc_gradient_penalty(netD, real_data, fake_data, device):
    alpha = torch.rand(real_data.size(dim=0), 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())

    ones = ones.to(device)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty


def train(tgt_model, optimizer, src_model, src_optimizer, dis_model, dis_optimizer, train_src_loader, train_tgt_loader,
          criterion, device, lr_e, lr_ac, lr_c, log_interval=100):
    tgt_model.train()
    dis_model0, dis_model1 = dis_model
    dis_optimizer0, dis_optimizer1 = dis_optimizer
    dis_model0.train()
    dis_model1.train()
    total_loss = 0
    src_loader = tqdm.tqdm(train_src_loader, smoothing=0, mininterval=1.0)
    tgt_loader = tqdm.tqdm(train_tgt_loader, smoothing=0, mininterval=1.0)
    total_dis_loss = 0.0
    total_tgt_loss = 0.0
    n_critic_count = 0
    num_dis = 0
    num_tgt = 0
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1
    one.to(device)
    mone.to(device)
    for p in src_model.parameters():
        p.requires_grad = False
    src_model.eval()
    for p in tgt_model.parameters():
        p.requires_grad = True

    for i, data_pack in enumerate(zip(src_loader, tgt_loader)):
        src_pack, tgt_pack = data_pack
        s_categorical_fields, s_numerical_fields, s_labels = src_pack
        t_categorical_fields, t_numerical_fields, t_labels = tgt_pack
        s_categorical_fields, s_numerical_fields, s_labels = s_categorical_fields.to(device), s_numerical_fields.to(
            device), s_labels.to(device)
        t_categorical_fields, t_numerical_fields, t_labels = t_categorical_fields.to(device), t_numerical_fields.to(
            device), t_labels.to(device)
        if s_categorical_fields.size(dim=0) != t_categorical_fields.size(dim=0):
            break

        for p in dis_model0.parameters():
            p.requires_grad = True

        dis_model0.zero_grad()

        for p in dis_model1.parameters():
            p.requires_grad = True

        dis_model1.zero_grad()

        with torch.no_grad():
            _, src_feat = src_model(s_categorical_fields, s_numerical_fields)
        if isinstance(src_feat, list):
            src_feat = torch.cat(src_feat, dim=1)

        ### extension
        tgt_model.eval()
        with torch.no_grad():
            _, src_feat1 = tgt_model(s_categorical_fields, s_numerical_fields)
        tgt_model.train()
        if isinstance(src_feat1, list):
            src_feat1 = torch.cat(src_feat1, dim=1)
        ##

        criticD_real0 = dis_model0(src_feat.detach())
        criticD_real0 = criticD_real0.mean()
        criticD_real0.backward(mone)

        criticD_real1 = dis_model1(src_feat1.detach())
        criticD_real1 = criticD_real1.mean()
        criticD_real1.backward(mone)

        criticD_real0 = dis_model0(src_feat1.detach())
        criticD_real0 = criticD_real0.mean()
        criticD_real0.backward(mone)
        
        criticD_real1 = dis_model1(src_feat.detach())
        criticD_real1 = criticD_real1.mean()
        criticD_real1.backward(mone)

        _, tgt_feat = tgt_model(t_categorical_fields, t_numerical_fields)
        if isinstance(tgt_feat, list):
            tgt_feat = torch.cat(tgt_feat, dim=1)

        ##EXTENSION

        criticD_fake0 = dis_model0(tgt_feat.detach())
        criticD_fake0 = criticD_fake0.mean()
        criticD_fake0.backward(one)

        gradient_penalty0 = calc_gradient_penalty(dis_model0, src_feat.data, tgt_feat.data, device)
        gradient_penalty0.backward()

        dis_optimizer0.step()

        #########
        criticD_fake1 = dis_model1(tgt_feat.detach())
        criticD_fake1 = criticD_fake1.mean()
        criticD_fake1.backward(one)

        gradient_penalty1 = calc_gradient_penalty(dis_model1, src_feat1.data, tgt_feat.data, device)
        gradient_penalty1.backward()
        dis_optimizer1.step()

        num_dis += 1
        n_critic_count += 1

        if n_critic_count >= 5:
            for p in dis_model0.parameters():
                p.requires_grad = False

            for p in dis_model1.parameters():
                p.requires_grad = False

            tgt_model.zero_grad()
            tgt_y, tgt_feat = tgt_model(t_categorical_fields, t_numerical_fields)
            if isinstance(tgt_feat, list):
                tgt_feat = torch.cat(tgt_feat, dim=1)
            ##EXTENSION
            criticG_fake0 = dis_model0(tgt_feat)
            criticG_fake0 = criticG_fake0.mean()
            G_cost = -criticG_fake0

            criticG_fake1 = dis_model1(tgt_feat)
            criticG_fake1 = criticG_fake1.mean()
            G_cost -= criticG_fake1

            loss_list = [criterion(tgt_y[i], t_labels[:, i].float()) for i in range(t_labels.size(1))]
            loss = 0
            loss += loss_list[0]*lr_c
            loss += loss_list[1]*lr_ac
            loss /= len(loss_list)

            loss += lr_e*G_cost
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tgt_loader.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
            n_critic_count = 0
        
    print(total_dis_loss/num_dis)


def metatrain(model, optimizer, data_loader, device, log_interval=100):
    model.train()
    total_loss = 0
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
            device), labels.to(device)
        batch_size = int(categorical_fields.size(0) / 2)
        list_sup_categorical.append(categorical_fields[:batch_size])
        list_qry_categorical.append(categorical_fields[batch_size:])
        list_sup_numerical.append(numerical_fields[:batch_size])
        list_qry_numerical.append(numerical_fields[batch_size:])
        list_sup_y.append(labels[:batch_size])
        list_qry_y.append(labels[batch_size:])

        if (i + 1) % 2 == 0:
            loss = model.global_update(list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical,
                                       list_qry_numerical, list_qry_y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            list_sup_categorical, list_sup_numerical, list_sup_y, list_qry_categorical, list_qry_numerical, list_qry_y = list(), list(), list(), list(), list(), list()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, task_num, device):
    model.eval()
    labels_dict, predicts_dict, loss_dict = {}, {}, {}
    for i in range(task_num):
        labels_dict[i], predicts_dict[i], loss_dict[i] = list(), list(), list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(
                device), labels.to(device)
            y, _ = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                labels_dict[i].extend(labels[:, i].tolist())
                predicts_dict[i].extend(y[i].tolist())
                loss_dict[i].extend(
                    torch.nn.functional.binary_cross_entropy(y[i], labels[:, i].float(), reduction='none').tolist())
    auc_results, loss_results = list(), list()
    for i in range(task_num):
        auc_results.append(roc_auc_score(labels_dict[i], predicts_dict[i]))
        loss_results.append(np.array(loss_dict[i]).mean())
    return auc_results, loss_results


import random


def main(tgt_dataset_name,
         src_dataset_name,
         dataset_path,
         task_num,
         expert_num,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         device,
         save_dir,
         read_dir,
         chunksize,
         learning_rateda,
         lr_e,
         lr_ac,
         lr_c):
    device = torch.device(device)
    field_dims = {
        'AliExpress_NL': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_ES': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_FR': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_US': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2],
        'AliExpress_RU': [11, 4, 7, 2, 33, 7, 50, 8, 8, 2, 2, 2, 2, 2, 2, 2]
    }

    save_path = f'{save_dir}/{src_dataset_name}_{tgt_dataset_name.split("_")[-1]}_{model_name}_{time.strftime("%m%d_%H%M%S")}.pt'
    print(save_path)
    src_read_path = f'{read_dir}/{src_dataset_name}_{model_name}.pt'
    tgt_read_path = f'{read_dir}/{src_dataset_name}_{model_name}.pt'

    if model_name in ['sharedbottom', 'omoe']:
        dis_model0 = Discriminator(input_dims=256, hidden_dims=64).to(device)
        dis_model1 = Discriminator(input_dims=256, hidden_dims=64).to(device)
    else:
        dis_model0 = Discriminator(input_dims=256 * task_num, hidden_dims=64).to(device)
        dis_model1 = Discriminator(input_dims=256 * task_num, hidden_dims=64).to(device)

    dis_optimizer0 = torch.optim.Adam(params=dis_model0.parameters(), lr=learning_rateda, weight_decay=weight_decay)
    ##dual
    dis_optimizer1 = torch.optim.Adam(params=dis_model1.parameters(), lr=learning_rateda, weight_decay=weight_decay)
    ##
    dis_model = [dis_model0, dis_model1]
    dis_optimizer = [dis_optimizer0, dis_optimizer1]

    criterion = torch.nn.BCELoss()
    early_stopper = EarlyStopper(num_trials=5, save_path=save_path)
    tgt_field_dims = field_dims[tgt_dataset_name]
    src_field_dims = field_dims[src_dataset_name]
    tgt_numerical_num, src_numerical_num = 63, 63
    src_model = get_model(model_name, src_field_dims, src_numerical_num, task_num, expert_num, embed_dim).to(device)
    tgt_model = get_model(model_name, tgt_field_dims, tgt_numerical_num, task_num, expert_num, embed_dim).to(device)
    tgt_optimizer = torch.optim.Adam(params=tgt_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    src_optimizer = torch.optim.Adam(params=src_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    src_model.load_state_dict(torch.load(src_read_path,map_location=torch.device(args.device)))
    tgt_model.load_state_dict(torch.load(tgt_read_path,map_location=torch.device(args.device)))

    
    for epoch_i in range(epoch):
        tgt_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/split_train.csv',
                                chunksize=batch_size * chunksize)
        src_datas = pd.read_csv(os.path.join(dataset_path, src_dataset_name) + '/train.csv',
                                chunksize=batch_size * chunksize, skiprows=lambda i: i > 0 and random.random() > 0.5)
        test_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/test.csv',
                                 chunksize=batch_size * chunksize)
        for i, pack in enumerate(zip(tgt_datas, src_datas)):
            tgt_trainset, src_trainset = pack[0].to_numpy()[:, 1:], pack[1].to_numpy()[:, 1:]
            tgt_trainset = AliExpressDataset(tgt_trainset)
            src_trainset = AliExpressDataset(src_trainset)
            train_src_loader = DataLoader(src_trainset, batch_size=batch_size, num_workers=8, shuffle=True)
            train_tgt_loader = DataLoader(tgt_trainset, batch_size=batch_size, num_workers=8, shuffle=True)
            if model_name == 'metaheac':
                metatrain(tgt_model, tgt_optimizer, train_tgt_loader, device)
            else:
                train(tgt_model, tgt_optimizer, src_model, src_optimizer, dis_model, dis_optimizer, train_src_loader,
                      train_tgt_loader, criterion, device, lr_e, lr_ac, lr_c)

        # epoch evaluate
        auc_results, loss_results = [], []
        for i, test_data in enumerate(test_datas):
            test_data = test_data.to_numpy()[:, 1:]
            test_data = AliExpressDataset(test_data)
            test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
            auc, loss = test(tgt_model, test_data_loader, task_num, device)
            auc_results.append(auc)
            loss_results.append(loss)
        auc_results, loss_results = np.array(auc_results), np.array(loss_results)
        aus_ans, loss_ans = [], []
        for k in range(task_num):
            aus_ans.append(np.mean(auc_results[:, k]))
            loss_ans.append(np.mean(loss_results[:, k]))

        print('epoch:', epoch_i, 'test: auc:', aus_ans)
        for i in range(task_num):
            print('task {}, AUC {}, Log-loss {}'.format(i, aus_ans[i], loss_ans[i]))

        if not early_stopper.is_continuable(tgt_model, np.array(aus_ans).mean()):
            print(f'test: best auc: {early_stopper.best_accuracy}')
            break

    tgt_model.load_state_dict(torch.load(save_path,map_location=torch.device(args.device)))
    auc_results, loss_results = [], []
    test_datas = pd.read_csv(os.path.join(dataset_path, tgt_dataset_name) + '/test.csv', chunksize=batch_size * chunksize)
    for i, test_data in enumerate(test_datas):
        test_data = test_data.to_numpy()[:, 1:]
        test_data = AliExpressDataset(test_data)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=False)
        auc, loss = test(tgt_model, test_data_loader, task_num, device)
        auc_results.append(auc)
        loss_results.append(loss)

    auc_results, loss_results = np.array(auc_results), np.array(loss_results)
    aus_ans, loss_ans = [], []
    for k in range(task_num):
        aus_ans.append(np.mean(auc_results[:, k]))
        loss_ans.append(np.mean(loss_results[:, k]))

    f = open('{}_{}.txt'.format(model_name, tgt_dataset_name), 'a', encoding='utf-8')
    f.write('main_da.py | {} | Time: {}\n'.format(model_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    f.write('Save Path: {}\n'.format(save_path))
    f.write('Source Domain: {}->{}\n'.format(src_dataset_name, tgt_dataset_name))
    f.write('learning rate: {}\n'.format(learning_rate))
    for i in range(task_num):
        print('task {}, AUC {}, Log-loss {}'.format(i, aus_ans[i], loss_ans[i]))
        f.write('task {}, AUC {}, Log-loss {}\n'.format(i, aus_ans[i], loss_ans[i]))

    print('\n')
    f.write('\n')
    f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_dataset_name', default='AliExpress_NL',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'AliExpress_RU'])
    parser.add_argument('--src_dataset_name', default='AliExpress_RU',
                        choices=['AliExpress_NL', 'AliExpress_ES', 'AliExpress_FR', 'AliExpress_US', 'AliExpress_RU'])
    parser.add_argument('--dataset_path', default='./data/')
    parser.add_argument('--model_name', default='mmoe',
                        choices=['singletask', 'sharedbottom', 'omoe', 'mmoe', 'ple', 'aitm', 'metaheac'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkptda')
    parser.add_argument('--read_dir', default='chkpt_src')
    parser.add_argument('--chunksize', type=int, default=1024)
    parser.add_argument('--learning_rateda', type=float, default=0.01)
    parser.add_argument('--lr_e', type=float, default=1.0)
    parser.add_argument('--lr_ac', type=float, default=1.0)
    parser.add_argument('--lr_c', type=float, default=1.0)
    args = parser.parse_args()
    main(args.tgt_dataset_name,
         args.src_dataset_name,
         args.dataset_path,
         args.task_num,
         args.expert_num,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.embed_dim,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.read_dir,
         args.chunksize,
         args.learning_rateda,
         args.lr_e,
         args.lr_ac,
         args.lr_c)