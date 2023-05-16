import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from argparse import ArgumentParser
from tqdm import tqdm
import shap
import pickle

def combine(dir, samples):
    npy_list = os.listdir(dir)
    npy_list = np.sort(np.array(npy_list))
    a = np.load(dir+npy_list[0]).flatten()
    rst = np.ones((len(samples),int(len(a)*3/4)))
    for i,npy in enumerate(tqdm(samples)):
        mtx = np.load(dir+str(npy)+".npy")[:3,:].T.flatten()
        rst[i] = mtx
    return rst

class VAE(pl.LightningModule):
    def __init__(self, in_dim, enc_out_dim=128, latent_dim=64, out_dim=1):
        super().__init__()
        self.save_hyperparameters()
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)
        # regression
        self.fr_mu = nn.Linear(enc_out_dim,out_dim)
        self.fr_var = nn.Linear(enc_out_dim,out_dim)
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.encode = nn.Sequential(
            nn.Linear(in_dim, enc_out_dim*4),
            nn.ReLU(),
            nn.Linear(enc_out_dim*4, enc_out_dim*2),
            nn.ReLU(),
            nn.Linear(enc_out_dim*2, enc_out_dim)
        )
        self.decode = nn.Sequential(
            nn.Linear(latent_dim, enc_out_dim),
            nn.ReLU(),
            nn.Linear(enc_out_dim, enc_out_dim*4),
            nn.ReLU(),
            nn.Linear(enc_out_dim*4, in_dim),
        )
    def forward(self,x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encode(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        r_mu, r_log_var = self.fr_mu(x_encoded), self.fr_var(x_encoded)
        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        # sample c from q_c
        std_c = torch.exp(r_log_var / 2)
        q_c = torch.distributions.Normal(r_mu, std_c)
        c = q_c.rsample()
        return c

class MyDataset(data.Dataset):
    def __init__(self, folder, label_table="../ukb_icd10/AFR.train.icd10.eid", trait='D64'):
        # npy_list = os.listdir(folder)
        label_list = label_table['ID'].values
        self.npy_list = np.core.defchararray.add(label_list.astype(str), np.repeat(".npy", len(label_list)))
        self.folder = folder
        # for subject in npy_list:
        #     id = subject.split('.')[0]
        self.labels = label_table[['ID',trait]]
        self.trait = trait



    def __getitem__(self, index):
        
        npy_name = self.npy_list[index]
        id = npy_name.split('.')[0]
        # print(id)
        # print(self.labels)
        y = self.labels.loc[self.labels['ID']==int(id)][self.trait].values

        x = np.load(self.folder+'/'+npy_name)[:3,:]
        x = x.T.flatten()
        # print(x.shape)
        return torch.from_numpy(x).to(torch.float32), y

    def __len__(self):
        return len(self.npy_list)

parser = ArgumentParser()
parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--trait', type=str, default='red_blood_cell_count', help='trait of interest')
parser.add_argument('--feature', type=str, default='t100k', help='type of feature')

args=parser.parse_args()

# make a dataloader 
home = "/proj/yunligrp/users/xiaoqi/prs_dl/data/"
test_dir = home + args.trait+"/npy_ukb_EUR_"+args.trait+"_test."+ args.feature+ "/processed/full_inds/full_chrs/encoded_outputs/"
test_pheno_dir = "/proj/yunligrp/users/xiaoqi/prs_dl/data/pheno/ukb_EUR_pheno_test.regenie.tsv"

y_te = pd.read_csv(test_pheno_dir, sep='\t').dropna()
# [["ID",args.trait]]
# train_dataset = MyDataset(train_dir, y_tr, trait = args.trait)
test_dataset = MyDataset(test_dir, y_te, trait = args.trait) 

# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4) # modify batch_size if necessary
print('Finished loading data.')


# train_mtx = torch.tensor(combine(train_dir, y_tr.index), dtype=torch.float32)
test_mtx = torch.tensor(combine(test_dir, y_te["ID"].values), dtype=torch.float32)
# test_mtx = combine(test_dir, y_te["ID"].values)

len_trset=355620
vae = VAE.load_from_checkpoint('../output/models/ukb_EUR_'+args.trait +'_model.vae.'+ args.feature+ '.' + str(len_trset)+ '.pth',in_dim=test_dataset[0][0].shape[0])
# en = pickle.load(open('../output/models/ukb_EUR_'+args.trait +'_model.en.'+ args.feature + '.'+ str(len_trset) + '.pkl','rb'))
print('Finished loading model.')

## VAE
e = shap.DeepExplainer(vae, test_mtx[:100])
shap_values = e.shap_values(test_mtx[:50])
shap_interaction = e.shap_interaction_values(test_mtx[:50])

## EN
# e = shap.Explainer(en, test_mtx[:100])
# shap_values = e.shap_values(test_mtx[:50])
# shap_interaction = e.shap_interaction_values(test_mtx[:50])

print('Finished SHAP.')
np.savetxt("../output/shap/ukb_EUR_"+args.trait +"_weights.vae." + args.feature + ".txt", shap_values)
np.savetxt("../output/shap/ukb_EUR_"+args.trait +"_weights.vae." + args.feature + ".interaction.txt", shap_interaction)

# np.savetxt("../output/shap/ukb_EUR_"+args.trait +"_weights.en." + args.feature + ".txt", shap_values)
# np.savetxt("../output/shap/ukb_EUR_"+args.trait +"_weights.en." + args.feature + ".interaction.txt", shap_interaction)