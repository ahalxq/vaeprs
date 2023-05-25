import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import random_split
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser

pl.seed_everything(1234)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad1d(reflection_padding)
        self.conv2d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class VAE(pl.LightningModule):
    def __init__(self, in_dim, enc_out_dim=128, latent_dim=64, out_dim=1, alpha = 0.7, l1_lambda=0.05, l2_lambda=0.05):
        super().__init__()

        self.save_hyperparameters()

        # distribution parameters
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
        self.alpha = alpha
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-6)

    def l1_reg(self):
        l1_norm = self.fr_mu.weight.abs().sum() + self.fr_var.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.fr_mu.weight.pow(2).sum() + self.fr_var.weight.pow(2).sum()
        
        return self.l2_lambda * l2_norm

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, y = batch

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

        # decoded
        x_hat = self.decode(z)

        # reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # label loss
        label_loss = F.mse_loss(c,y.float())
        # label_loss = torch.div(0.5*torch.square(r_mu-y),torch.exp(r_log_var)) + 0.5*r_log_var
        # elbo + self.l1_reg() + self.l2_reg()
        elbo = ((1-self.alpha)*(kl + recon_loss) + self.alpha*label_loss) 
        # elbo = (kl + recon_loss + label_loss ) 
        elbo = elbo.mean()

        self.log_dict({
            'train_loss': elbo,
            'train_recon_loss': recon_loss.mean(),
            'train_kl': kl.mean(),
            'train_label_loss': label_loss.mean()
        })

        return elbo

    def validation_step(self, batch, batch_idx):
        x, y = batch

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

        # decoded
        x_hat = self.decode(z)

        # reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # label loss
        label_loss = F.mse_loss(c,y.float())
        # elbo
        elbo = ((1-self.alpha)*(kl + recon_loss) + self.alpha*label_loss)
        # elbo = (kl + recon_loss + label_loss ) 
        elbo = elbo.mean()

        self.log_dict({
            'val_loss': elbo,
            'val_recon_loss': recon_loss.mean(),
            'val_kl': kl.mean(),
            'val_label_loss': label_loss.mean()
        })

        return elbo

    def test_step(self, batch, batch_idx):
        x, y = batch

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

        # decoded
        x_hat = self.decode(z)

        # reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # label loss
        label_loss = F.mse_loss(c,y.float())
        # elbo
        elbo = ((1-self.alpha)*(kl + recon_loss) + self.alpha*label_loss)
        # elbo = (kl + recon_loss + label_loss ) 
        elbo = elbo.mean()

        return elbo


# a helper class to load data, keep it as is
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
        x = np.argmax(x, axis=0)
        x = x.T.flatten()
        return torch.from_numpy(x).to(torch.float32), y

    def __len__(self):
        return len(self.npy_list)


def train():
    parser = ArgumentParser(description="This script uses variational autoencoder based regression model to predict polygenic quantitative traits.")
    parser.add_argument('--gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--trait', type=str, default='example', help='trait of interest')
    parser.add_argument('--feature', type=str, default='100', help='type of feature')
    parser.add_argument('--samples', type=int, default=None, help='training sample size')
    parser.add_argument('--alpha', type=float, default=0.7, help='weight on label loss')
    parser.add_argument('--home', type=str, default="./vaeprs/simulated/", help='Input directory')
    parser.add_argument('--outdir', type=str, default="./vaeprs/simulated/", help='Output directory')

    args=parser.parse_args()

    # make a dataloader 
    train_dir = args.home + args.trait+"/npy_"+args.trait+"_train."+ args.feature+ "/processed/full_inds/full_chrs/encoded_outputs/"
    train_pheno_dir = args.home + "pheno_train.regenie.tsv"
    test_dir = args.home + args.trait+"/npy_"+args.trait+"_test."+ args.feature+ "/processed/full_inds/full_chrs/encoded_outputs/"
    test_pheno_dir = args.home + "pheno_test.regenie.kinship.rm.tsv"

    y_tr = pd.read_csv(train_pheno_dir, sep='\t').dropna()
    y_te = pd.read_csv(test_pheno_dir, sep='\t')[["ID",args.trait]].dropna()

    if not args.samples:
        y_tr = pd.read_csv(train_pheno_dir, sep='\t').dropna()
        num_samples = y_tr.shape[0]
    else:
        train_id = args.home + args.trait+"."+ str(args.samples) + ".train.id"
        ids = pd.read_csv(train_id, sep='\t', header = 0, names = ["ID"]).dropna()
        tmp = pd.read_csv(train_pheno_dir, sep='\t').dropna()
        y_tr = tmp.merge(ids, on="ID")
        num_samples = y_tr.shape[0]

    train_dataset = MyDataset(train_dir, y_tr, trait = args.trait)
    test_dataset = MyDataset(test_dir, y_te, trait = args.trait) 

    
    # if not args.samples:
    len_trset = len(train_dataset)
    val_size = len_trset//10
    train_set, val_set = random_split(train_dataset,[len_trset-val_size ,val_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4) # modify batch_size if necessary
    print('Finished loading data.')

    vae = VAE(train_dataset[0][0].shape[0], alpha = args.alpha)
    es = EarlyStopping(monitor='val_label_loss',patience=8)
    wandb_logger = WandbLogger(''+args.trait + '_'+ args.feature, project='PRS_DL')  
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=200, callbacks=[es], logger=wandb_logger)
    trainer.fit(vae, train_loader, val_loader)


    print('model training is done.')
    # save your model
    trainer.save_checkpoint(args.outdir+args.trait +'_model.vae.'+ args.feature +'.pth')  
    vae = VAE.load_from_checkpoint(args.outdir+args.trait +'_model.vae.'+ args.feature+'.pth',in_dim=train_dataset[0][0].shape[0])
    trainer.test(vae,test_loader)

    r2=None
    ry=None
    rvar=None
    with torch.no_grad():
        for (x,y) in test_loader:
            # x, y = x.cuda(), y.cuda()
            vae.eval()
            output = vae.encode(x)
            pred = vae.fr_mu(output)
            var = vae.fr_var(output)
            if r2 == None:
                r2 = pred
                ry = y
                rvar = var
            else:
                r2 = torch.cat((r2,pred))
                ry = torch.cat((ry,y))
                rvar = torch.cat((rvar,var))
    corr, _ = pearsonr(ry.squeeze(),r2.squeeze())
    print('r2 = {}'.format(r2_score(ry,r2)))
    print('PCC = {}'.format(corr))
    print('MSE = {}'.format(mean_squared_error(ry,r2)))

if __name__ == '__main__':
    train()