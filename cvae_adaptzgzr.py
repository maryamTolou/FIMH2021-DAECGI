import os
import sys
import argparse
import numpy as np
import scipy.io
import csv
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pybobyqa
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import pybobyqa
from numpy import linalg as LA
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--latent', type=int, default=8, help='latent dimension')
parser.add_argument('--epoch', type=int, default=2200, help='epoch for VAE')
parser.add_argument('--pre_train', type=int, default=0, help='0 for new model, 1 for pre-train')
args = parser.parse_args()
# num_worker < physical cpu kernels

# Hyperparameters
BATCH_SIZE = 16
W = 1786
H = 120
lr = 1e-4
# bb = 'move'
# b1_ = 0.005
b1_ = 0.001
b2_ = 0
# Prepare dataset
# Prepare dataset
H_train_path = '/home/mt6129/H/final-data/train_sub/'
H_test_path = '/home/mt6129/H/final-data/test_sub/'
g_train_path = '/home/mt6129/H/final-data/train_geo_sub/'
g_test_path = '/home/mt6129/H/final-data/test_geo_sub/'
H_recon_path = '/home/mt6129/H/final-data/recon/'
g_recon_path = '/home/mt6129/H/final-data/recon_geo/'
# H_recon_path = '/home/mt6129/H/all_data/recon_H/'
# g_recon_path = '/home/mt6129/H/all_data/recon_g/'


def find_min_max(root_path, label_path):
    files = os.listdir(root_path)
    tot_data = len(files)
    xmin = 100
    xmax = -100
    gmin = 100
    gmax = -100
    for i, file in enumerate(files):
        print(i)
        print(file)
        path = os.path.join(root_path, file)
        matFile = scipy.io.loadmat(path)
        x = matFile['h']
        # resize u
        # u = cv2.resize(u, (478, 118), interpolation=cv2.INTER_LINEAR)
        if x.min()< xmin:
            xmin = x.min()
        if x.max()> xmax:
            xmax = x.max()
            # print(xmax)
        # g_ind = file.find("glob")
        # label = file[g_ind:]
        label = file
        path_label = os.path.join(label_path, label)
        matlabel = scipy.io.loadmat(path_label)
        g = matlabel['geo']

        if g.min()< gmin:
            gmin = g.min()
        if g.max()> gmax:
            gmax = g.max()


    return xmin, xmax, gmin, gmax


xmin1, xmax1, gmin1, gmax1 = find_min_max(H_train_path, g_train_path)
xmin2, xmax2, gmin2, gmax2 = find_min_max(H_test_path, g_test_path)
# xmin3, xmax3, gmin3, gmax3 = find_min_max(H_test_path, g_test_path)

if xmin1 < xmin2:
    xmin = xmin1
else :
    xmin = xmin2
if gmin1 < gmin2:
    gmin = gmin1
else :
    gmin = gmin2
if xmax1 > xmax2:
    xmax = xmax1
else :
    xmax = xmax2
if gmax1 > gmax2:
    gmax = gmax1
else :
    gmax = gmax2
    
# xmin, xmax, gmin, gmax = find_min_max(H_recon_path, g_recon_path,  xmin, xmax, gmin, gmax)
# print(gmax)
# print(xmax)
# print(gmin)
# print(xmin)

# exit()
class CustData(Dataset):
    def __init__(self, H_path, g_path):
        # self.root_path = root_path
        # print("in cuuuuuuuuuuuuuuust")
        self.H_pth = H_path
        self.g_path = g_path
        self.data = os.listdir(H_path)
        self.labels = os.listdir(g_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        H = self.data[idx]
        # print(H)
        name = H
        d_path = os.path.join(self.H_pth, H)
        # print(d_path)
        matFile = scipy.io.loadmat(d_path)
        x = matFile['h']
        # resize u
        # u = cv2.resize(u, (478, 118), interpolation=cv2.INTER_LINEAR)
        normalized = (x-xmin)/(xmax-xmin)
        x = torch.from_numpy(normalized).float()
        # x = torch.from_numpy(x).float()

        G = self.labels[idx]    
        g_path = os.path.join(self.g_path, G)
        matFile = scipy.io.loadmat(g_path)
        y = matFile['geo']
        # resize u
        # u = cv2.resize(u, (478, 118), interpolation=cv2.INTER_LINEAR)
        normalized = (y-gmin)/(gmax-gmin)
        y = torch.from_numpy(normalized).float()
        # y = torch.from_numpy(y).float()
        return x , y, H

# recon_test_data = scipy.io.loadmat('/home/mt6129/H/inf-dense/H1786.mat')
# recon_lab = scipy.io.loadmat('/home/mt6129/H/inf-dense/swi0vert0rot0_grtruth.mat')
# u_rec = recon_test_data['h']
# u_rec_lab = recon_lab['geo']
# normalized = (u_rec-u_rec.min())/(u_rec.max()-u_rec.min())
# lab_normalized = (u_rec_lab-u_rec_lab.min())/(u_rec_lab.max()-u_rec_lab.min())
# recon_list = np.zeros((1, 2, H, W))
# recon_lab_list = np.zeros((1, 1, H, W))
# recon_list[0, 0, :, :] = normalized
# recon_lab_list[0, 0, :, :] = lab_normalized
# recon_list[0, 1, :, :] = lab_normalized
# recon_set = np.array(recon_list)
# recon_lab_set = np.array(recon_lab_list)


train_dataset = CustData(H_train_path, g_train_path)
test_dataset = CustData(H_test_path, g_test_path)
recon_dataset = CustData(H_recon_path, g_recon_path)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
recon_loader = DataLoader(recon_dataset)



# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 15), stride=(2, 5), padding=(1, 7))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(2, 3), padding=(1, 2))
        # torch.nn.init.xavier_uniform_(self.conv1.weight, 1)
        # self.b1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2) , padding=(1,0))
        # torch.nn.init.xavier_uniform_(self.conv2.weight, 1)
        # self.b2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=(2, 2) , padding=(1, 0))
        self.conv5 = nn.Conv2d(256, 64, kernel_size=(4, 4), stride=(1, 2) , padding=(0, 0))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv7 = nn.Conv2d(32, 16, kernel_size=(1, 1))
        self.conv8 = nn.Conv2d(16, 8, kernel_size=(1, 1))
        self.conv91 = nn.Conv2d(8, latent_dim, kernel_size=1)
        # self.b51 = nn.BatchNorm2d(256)
        self.conv92 = nn.Conv2d(8, latent_dim, kernel_size=1)

        # # # best extension more layers so far ---- name:m = 'model_CNN_xj_move_16_1000' not any improvements. similar to the other
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(9, 17), stride=(3, 5), padding=(4, 8))
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 11), stride=(2, 5), padding=(1, 5))
        # # torch.nn.init.xavier_uniform_(self.conv1.weight, 1)
        # # self.b1 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 3) , padding=(1,2))
        # # torch.nn.init.xavier_uniform_(self.conv2.weight, 1)
        # # self.b2 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2) , padding=(1, 1))
        # self.conv5 = nn.Conv2d(256, 64, kernel_size=(1, 1))
        # self.conv6 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        # self.conv7 = nn.Conv2d(32, 16, kernel_size=(1, 1))
        # self.conv8 = nn.Conv2d(16, 8, kernel_size=(1, 1))
        # self.conv91 = nn.Conv2d(8, latent_dim, kernel_size=1)
        # # self.b51 = nn.BatchNorm2d(256)
        # self.conv92 = nn.Conv2d(8, latent_dim, kernel_size=1)

        #strong decoder
        self.dconv9 = nn.ConvTranspose2d(2*latent_dim, 16, kernel_size= 1)
        # self.db6= nn.BatchNorm2d(128)
        self.dconv8 = nn.ConvTranspose2d(16, 32, kernel_size= 1)
        self.dconv7 = nn.ConvTranspose2d(32, 64, kernel_size= 3 , stride=(1, 1) , padding=(1, 1))
        # self.dconv72 = nn.ConvTranspose2d(64, 96, kernel_size= 1 , stride=(1, 1))
        self.dconv6 = nn.ConvTranspose2d(64, 128, kernel_size= 3, stride=(1, 1) , padding=(1, 1))
        self.dconv62 = nn.ConvTranspose2d(128, 256, kernel_size= 3, stride=(1, 1) , padding=(1, 1))
        self.dconv5 = nn.ConvTranspose2d(256, 128,  kernel_size=(3, 3), stride=(2, 2) , padding=(0, 0))
        # self.db5 = nn.BatchNorm2d(64)
        self.dconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 3) , padding=(1,2))
        # self.db4= nn.BatchNorm2d(32)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 7), stride=(2, 5), padding=(3, 0))
        # self.db3 = nn.BatchNorm2d(16)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(10, 14), stride=(3, 5), padding=(2, 1))
        # self.db2 = nn.BatchNorm2d(8)
        self.dconv1 = nn.ConvTranspose2d(16, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 3))
        self.dconv12 = nn.ConvTranspose2d(8, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # self.db1 = nn.BatchNorm2d(1)
        # torch.nn.init.xavier_uniform_(self.dconv1.weight, 1)


        # #weak decoder
        # self.dconv9 = nn.ConvTranspose2d(2*latent_dim, 32, kernel_size= 1)
        # self.dconv8 = nn.ConvTranspose2d(32, 16, kernel_size= (3, 5), stride=(2, 3))
        # self.dconv7 = nn.ConvTranspose2d(16, 8, kernel_size= (5, 7) , stride=(3, 3))
        # self.dconv6 = nn.ConvTranspose2d(8, 4, kernel_size= 11 , stride=(3, 5))
        # self.dconv5 = nn.ConvTranspose2d(4, 1, kernel_size= (8, 5) , stride=(1, 3), padding=(0, 2))

        ### second encoder
        self.conv12 = nn.Conv2d(1, 16, kernel_size=(9, 17), stride=(3, 5), padding=(4, 8))
        # torch.nn.init.xavier_uniform_(self.conv1.weight, 1)
        # self.b1 = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(16, 32, kernel_size=(5, 11), stride=(2, 5), padding=(1, 5))
        # torch.nn.init.xavier_uniform_(self.conv2.weight, 1)
        # self.b2 = nn.BatchNorm2d(64)
        self.conv32 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2, 3) , padding=(1,2))
        # torch.nn.init.xavier_uniform_(self.conv3.weight, 1)
        # self.b3 = nn.BatchNorm2d(128)
        self.conv42 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2) , padding=(1, 1))
        # torch.nn.init.xavier_uniform_(self.conv4.weight, 1)
        # self.b4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(128, 128, kernel_size=(1, 1))
        self.conv52 = nn.Conv2d(128, 64, kernel_size=(1, 1))
        self.conv62 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv72 = nn.Conv2d(32, 8, kernel_size=(1, 1))
        self.conv921 = nn.Conv2d(8, latent_dim, kernel_size=1)
        self.conv922 = nn.Conv2d(8, latent_dim, kernel_size=1)
    
        # self.conv12 = nn.Conv2d(1, 32, kernel_size=(5, 15), stride=(2, 5), padding=(1, 7))
        # self.conv22 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(2, 3), padding=(1, 2))
        # # torch.nn.init.xavier_uniform_(self.conv1.weight, 1)
        # # self.b1 = nn.BatchNorm2d(32)
        # self.conv32 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2) , padding=(1,0))
        # # torch.nn.init.xavier_uniform_(self.conv2.weight, 1)
        # # self.b2 = nn.BatchNorm2d(64)
        # self.conv42 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=(2, 2) , padding=(1, 0))
        # self.conv52 = nn.Conv2d(256, 64, kernel_size=(4, 4), stride=(1, 2) , padding=(0, 0))
        # self.conv62 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        # self.conv72 = nn.Conv2d(32, 16, kernel_size=(1, 1))
        # self.conv82 = nn.Conv2d(16, 8, kernel_size=(1, 1))
        # self.conv921 = nn.Conv2d(8, latent_dim, kernel_size=1)
        # # self.b51 = nn.BatchNorm2d(256)
        # self.conv922 = nn.Conv2d(8, latent_dim, kernel_size=1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def encode(self, data):
        x = F.elu(self.conv1(data))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.elu(self.conv7(x))
        x = F.elu(self.conv8(x))
        mu = F.elu(self.conv91(x))
        # print(mu.size())
        logvar = F.elu(self.conv92(x))
        # print("data in endcode", str(mu.shape))
        # print("siiiiiize")
        return mu, logvar
    
    def encode2(self, data):
        x = F.elu(self.conv12(data))
        x = F.elu(self.conv22(x))
        x = F.elu(self.conv32(x))
        x = F.elu(self.conv42(x))
        x = F.elu(self.conv52(x))
        x = F.elu(self.conv62(x))
        x = F.elu(self.conv72(x))
        # x = F.elu(self.conv82(x))
        mu = F.elu(self.conv921(x))
        logvar = F.elu(self.conv922(x))
        # print("here in endcode", str(z.shape))
        # print(mu.size())
        # print("siiiiiize")
        return mu, logvar
    

    def decode(self, z):
        x = F.elu(self.dconv9(z))
        # # print(x.size())
        #  + self.dconv9(z)
        x = F.elu(self.dconv8(x))
        # self.dconv9(z)
        x = F.elu(self.dconv7(x))
        # print(x.size())
        # + self.dconv9(z)
        x = F.elu(self.dconv6(x))
        x = F.elu(self.dconv62(x))
        # # print(x.size())
        #  + self.dconv9(z)
        x = F.elu(self.dconv5(x))
        # # print(x.size())
        x = F.elu(self.dconv4(x))
        # # print(x.size())
        x = F.elu(self.dconv3(x))
        # # print(x.size())
        x = F.elu(self.dconv2(x))
        # # print(x.size())
        x = F.elu(self.dconv1(x))
        x = F.elu(self.dconv12(x))
        # print(x.size())
        # x = self.db1(x)
        return x

    #with skip
    # def decode2(self, z):
    #     x1 = F.elu(self.dconv9(z))
    #     # # print(x.size())
    #     #  + self.dconv9(z)
    #     x2 = F.elu(torxh.cat(self.dconv8(x1), self.dconv9(z))
    #     # self.dconv9(z)
    #     x3 = F.elu(self.dconv7(x2), self.dconv8(x1))
    #     # print(x.size())
    #     # + self.dconv9(z)
    #     x4 = F.elu(self.dconv6(x3), self.dconv7(x2))
    #     x5 = F.elu(self.dconv62(x4), self.dconv7(x3))
    #     # # print(x.size())
    #     #  + self.dconv9(z)
    #     x6 = F.elu(self.dconv5(x5), self.dconv62(x4))
    #     # # print(x.size())
    #     x7 = F.elu(self.dconv4(x6), self.dconv5(x5))
    #     # # print(x.size())
    #     x8 = F.elu(self.dconv3(x7), self.dconv4(x6))
    #     # # print(x.size())
    #     x9 = F.elu(self.dconv2(x8), self.dconv3(x7))
    #     # # print(x.size())
    #     x10 = F.elu(self.dconv1(x9), self.dconv2(x8))
    #     x11 = F.elu(self.dconv12(x10), self.dconv1(x9))
    #     # print(x.size())
    #     # x = self.db1(x)
    #     return x
    
    def forward(self, data, label, name):
        N, h, w = data.shape
        # print(N)
        # print(h)
        # print(w)
        # exit()
        data = data.view(N, -1, h, w)
        label = label.view(N, -1, h, w)
        mu, logvar = self.encode(data)
        mu2, logvar2 = self.encode2(label)
        # print(z2.shape)
        # print("uppppppp")
        zr = self.reparameterize(mu, logvar)
        z2 = self.reparameterize(mu2, logvar2)
        # print(z.shape)
        z = torch.cat((zr, z2),  dim=1)
        # print(z.shape)
        # print("khaaaaaaaaaaaaaaaaaaar")
        # exit()
        u = self.decode(z)
        u = u.view(N, h, w)
        return u, mu, logvar, mu2, logvar2, name

    def generate(self):
        n = 4
        temp_ = torch.randn(n, 8, 5, 13)
        # print(temp_)
        z = torch.rand_like(temp_).to(device)
        x_gen = self.decode(z)
        # print("size")
        print(x_gen.size())
        return x_gen

    def generate_onlygeo(self, label, names):
        n = 16
        temp_ = torch.zeros(n, 8, 5, 12)
        # print(temp_)
        zr = torch.rand_like(temp_).to(device)
        label = label.view(n, -1, H, W)
        mu, logvar = self.encode2(label)
        z2 = self.reparameterize(mu, logvar)
        z = torch.cat((zr, z2),  dim=1)
        u = self.decode(z)
        u = u.view(n, H, W)
        # print("size")
        # print(x_gen.size())
        return u, names

    def generate_lab(self, label):
        n = 1
        temp_ = torch.randn(1, 8, 5, 12)
        # print(temp_)
        zr = torch.rand_like(temp_).to(device)
        label = label.view(1, -1, H, W)
        mu, logvar = self.encode2(label)
        z2 = self.reparameterize(mu, logvar)
        z = torch.cat((zr, z2),  dim=1)
        u = self.decode(z)
        u = u.view(n, H, W)
        # print("size")
        # print(x_gen.size())
        return z, u, zr, z2

    def generate_lab_(self, label, names):
        n = 16
        # zr = torch.randn(n, 8, 5, 12).to(device)
        zr = torch.randn(n, 8, 5, 12).to(device)
        # print(temp_)
        # zr = torch.rand_like(temp_).to(device)
        label = label.view(n, -1, H, W)
        mu2, logvar2 = self.encode2(label)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2),  dim=1)
        u = self.decode(z)
        u = u.view(n, H, W)
        # print("size")
        # print(x_gen.size())
        return u, names

    def generate_onegeo(self, label, names, flag):
        n = 16
        # zr = torch.randn(n, 8, 5, 12).to(device)
        zr = torch.randn(n, 8, 5, 12).to(device)
        if flag == 1:
            zr = zr + -5*torch.ones(n, 8, 5, 12).to(device)

        # print(temp_)
        # zr = torch.rand_like(temp_).to(device)
        label = label.view(n, -1, H, W)
        mu2, logvar2 = self.encode2(label)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2),  dim=1)
        u = self.decode(z)
        u = u.view(n, H, W)
        # print("size")
        # print(x_gen.size())
        return u, names

    def var_change(self, data, label, change, i):
        h, w = data.shape
        # print(h)
        data = data.view(1, -1, h, w)
        label = label.view(1, -1, h, w)
        mu, logvar = self.encode(data)
        mu2, logvar2 = self.encode2(label)
        # print(mu2[0][i])
        ch = torch.from_numpy(change*np.ones((5,12)))
        ch = ch.to(device)
        # for k in range(8):
        #     mu[0][k] = mu[0][k] + ch
        mu[0][i] = ch
        # print(mu[0][i])
        # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
        zr = self.reparameterize(mu, logvar)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2),  dim=1)
        u = self.decode(z)

        return u, mu, logvar
        
    def call_dec(self, zh):
        # z_nump = torch.from_numpy(zh).to(device)
        # z_nump = z_nump.float()
        Hzp = self.decode(zh)

        return Hzp

def loss_function(recon_x, x, mu, logvar, mu2, logvar2, b, b2):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    return BCE, KLD, BCE + b*KLD + 0*KLD2, KLD2
    # return BCE, KLD, BCE +0*KLD + 0*KLD2, KLD2


model = VAE(args.latent)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists('img'):
    os.makedirs('img')
if args.pre_train:
    print('Load pre-trained')
    model_state = torch.load(os.path.join(model_dir, '_{}_{}_CVAE_'.format(args.latent, args.epoch)), map_location='cuda:0')
    model.load_state_dict(model_state)


def inline_print(s):
    sys.stdout.write(s + '\r')
    sys.stdout.flush()

# import pybobyqa
def train(epoch , b, b2):
    model.train()
    recon, latent, total, latent2 = 0, 0, 0, 0
    n = 0
    for x, y, name in train_loader:
        x = x.to(device)
        # print("khaaaaaaar")
        # print(x.shape)
        y = y.to(device)
        # x_ = torch.zeros((4, 120, 1786))
        # for i in range(4):
        #     x_[i] = x[i][0]
        # '/home/mt6129/H/checkdata-cvae'
        # print(x_[i].shape)
        # scipy.io.savemat('/home/mt6129/H/checkdata-cvae/check_data.mat', mdict={'data': x_[i].cpu().detach().numpy()})
        # x_ = x_.to(device)

        optimizer.zero_grad()
        u, mu, logvar, mu2, logvar2, name = model(x, y, name)
        
        recon_loss, latent_loss, total_loss, latent_loss2 = loss_function(u, x, mu, logvar, mu2, logvar2, b, b2)

        total_loss.backward()
        recon += recon_loss.item()
        latent += latent_loss.item()
        total += total_loss.item()
        latent2 +=latent_loss2.item()
        optimizer.step()

        n += 1
        log = '[E {:03d}] Loss: {:.4f}, Rec loss: {:.4f}, Latent loss: {:.4f},  Latent loss2: {:.4f}'.format(epoch, total / (n * BATCH_SIZE), recon / (n * BATCH_SIZE), latent / (n * BATCH_SIZE), latent2 / (n * BATCH_SIZE))
        inline_print(log)
    
    torch.save(model.state_dict(), model_dir + '/model_CNN_CVAE_rots_strong_decoder_newencoder_notnorm_{}_{}_b1_{}_b2_{}'.format(args.latent, args.epoch, b1_, b2_))
    return recon, latent, total, latent2


def test(b, b2):
    model.eval()
    recon, latent, total, latent2 = 0, 0, 0, 0
    with torch.no_grad():
        for x, y, names in test_loader:
            N, h, w = x.shape
            x = x.to(device)
            y = y.to(device)

            # x_ = torch.zeros((N, 120, 1786))
            # for i in range(N):
            #     x_[i] = x[i][0]

            # x_ = x_.to(device)

            u, mu, logvar, mu2, logvar2, names= model(x, y, names)
            recon_loss, latent_loss, total_loss, latent_loss2 = loss_function(u, x, mu, logvar, mu2, logvar2, b, b2)


            recon += recon_loss.item()
            latent += latent_loss.item()
            total += total_loss.item()
            latent2 +=latent_loss2.item()
    return recon, latent, total, latent2


def plot_losses(train_a, test_a, loss_type, num_epochs):
    """Plot epoch against train loss and test loss 
    """
    # plot of the train/validation error against num_epochs
    fig, ax1 = plt.subplots(figsize=(6, 5))
    ax1.set_xticks(np.arange(0 + 1, num_epochs + 1, step=10))
    ax1.set_xlabel('epochs')
    ax1.plot(train_a, color='green', ls='-', label='train {} loss'.format(loss_type))
    ax1.plot(test_a, color='red', ls='-', label='test {} loss'.format(loss_type))
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, fontsize='14', frameon=False)
    ax1.grid(linestyle='--')
    plt.tight_layout()
    fig.savefig('img/{}_{}.png'.format(loss_type, args.latent), dpi=300, bbox_inches='tight', transparent=True)
    plt.close()


def train_vae():
    recon_train_a, latent_train_a, total_train_a = [], [], []
    recon_test_a, latent_test_a, total_test_a = [], [], []
    # b = 0.005
    # 0.000001 too small
    #0.00001 kl loss small 0.1
    # b = 0.00003 best generation till now - recon loss about 0.04 and latent 161 and slowly decreasing - maybe try more epochs than 600 or increase beta a bit
    #also this b decreased to very small and then increased for a while and started decreasing slowly
    # b = 0.00005 I trained for 1500 epochs. the kl got to very small value of 0.002 and around epoch 600 it started increasing - epoch 640 kl is 164 . so I think if I use large values with more epochs thiis can help.
    # use larger than 0.00005 ------ !!!! it is decreasing after epoch 645 very slowly. after 1500 epochs I got to 104. use a larger b if good ok if not come back to values close to this with more epochs
    # tried b = 0.005. tyhe lat error got as small as 0.0001 and never started to increase in 1000 epochs. --> try smaller 1) either more epochs and b = 0.00005 2) a b in 0.00005 < b < 0.005
    # b = 0.0005 never increased after decreasing to values around 0.0001 after 1000 epochs. try smaller!
    # b = 0.00005 decreased and is again increasing but very slow!for 300 epochs it is still close to 104. choose a bigger b. slightly bigger!
    # b = 0.0001 converged to 65 after 1500 epochs so go bigger like 0.0005 and let it run for more than 1500 epochs or to get the range even try 0.001!
    b = 0.001
    b2 = 0
    file_name = "epch" + str(args.epoch) + "_b1_" + str(b) + "_b2_" + str(b2) + ".txt"
    with open('/home/mt6129/H/' + file_name, 'w+') as f:
        for epoch in range(args.epoch):
            # if epoch> 100:
            #     b = b + 0.001
            recon_train, latent_train, total_train, latent2 = train(epoch, b, b2)
            recon_test, latent_test, total_test, test_latent2 = test(b, b2)

            recon_train /= len(train_dataset)
            latent_train /= len(train_dataset)
            total_train /= len(train_dataset)
            latent2 /= len(train_dataset)
            recon_test /= len(test_dataset)
            latent_test /= len(test_dataset)
            total_test /= len(test_dataset)
            test_latent2 /= len(test_dataset)
            # log = '[Epoch {:03d}] Loss: {:.4f}, Recon loss: {:.4f}, Latent loss: {:.4f}'.format(epoch, total_test, recon_test, latent_test)
            # print(log)
            log = '[E {:03d}] Trn: {:.6f}, TrRec: {:.6f}, Trlt: {:.6f}, lt2: {:.4f}, Tst: {:.4f}, Tstrc: {:.6f}, Tstlt: {:.6f}, Tstl2: {:.4f}'.format(epoch , total_train, recon_train, latent_train, latent2, total_test,
                                                                                    recon_test, latent_test, test_latent2)
            print(log)
            f.writelines(log) 
            f.write("\n")
            recon_train_a.append(recon_train)
            latent_train_a.append(latent_train)
            total_train_a.append(total_train)
            recon_test_a.append(recon_test)
            latent_test_a.append(latent_test)
            total_test_a.append(total_test)
        
    plot_losses(recon_train_a, recon_test_a, 'reconstruction', args.epoch)
    plot_losses(latent_train_a, latent_test_a, 'latent', args.epoch)
    plot_losses(total_train_a, total_test_a, 'total', args.epoch)


# def reconstruction():
#     model.eval()
#     with torch.no_grad():
#         data = next(iter(test_loader))
#         data = data.to(device)
#         recon_data, _, _ = model(data)
    
#     # import ipdb; ipdb.set_trace()
#     # x_sample = data.view(BATCH_SIZE, -1)
#     x_sample = np.squeeze(data.detach().cpu().numpy())
#     print(x_sample.size)
#     # x_reconstruct = recon_data.view(BATCH_SIZE, -1)
#     x_reconstruct = np.squeeze(recon_data.detach().cpu().numpy())

#     n = np.sqrt(BATCH_SIZE).astype(np.int32)
#     print(n)
#     I_reconstructed = np.empty((H * n, 2 * W * n))
#     for i in range(n):
#         for j in range(n):
#             data = x_sample[i * n + j, :].reshape(H, W)
#             recon_data = x_reconstruct[i * n + j, :].reshape(H, W)
#             MSE_normal = F.mse_loss(recon_data, data)
#             print("normal recon loss with org =" + str(MSE_normal.item()))
#             x = np.concatenate(
#                 (x_reconstruct[i * n + j, :].reshape(H, W),
#                 x_sample[i * n + j, :].reshape(H, W)),
#                 axis=1
#             )
#             scipy.io.savemat('CNN_results/Mat/recon/' + '_xj_' + ' H_org_{}_{}.mat'.format(args.latent, args.epoch),
#                      mdict={'H_org': x_sample[i * n + j, :].reshape(H, W)})
#             scipy.io.savemat('CNN_results/Mat/recon/' +  'xj' + 'H_recon_{}_{}.mat'.format(args.latent, args.epoch),
#                      mdict={'H_recon': x_reconstruct[i * n + j, :].reshape(H, W)})
#             I_reconstructed[i * H:(i + 1) * H, j * 2 * W:(j + 1) * 2 * W] = x
#     fig = plt.figure()
#     plt.axis('off')
#     plt.tight_layout()
#     plt.imshow(I_reconstructed, cmap='gray')
#     plt.savefig('img/I_reconstructed_xj{}.png'.format(args.latent))
#     plt.close(fig)
#     MSE_normal = F.mse_loss(recon_data, data)
#     print("normal recon loss with org =" + str(MSE_normal.item()))



# def generation():
#     model.eval()
#     z = torch.randn(BATCH_SIZE, args.latent).to(device)
#     with torch.no_grad():
#         x_generated = model.decode(z)
#     x_generated = x_generated.data
#     x_generated = np.squeeze(x_generated.detach().cpu().numpy())
#     n = np.sqrt(BATCH_SIZE).astype(np.int32)
#     I_generated = np.empty((H * n, W * n))
#     for i in range(n):
#         for j in range(n):
#             I_generated[i * H:(i + 1)* H, j * W:(j + 1) * W] = x_generated[i*n+j, :].reshape(H, W)

#     fig = plt.figure()
#     plt.axis('off')
#     plt.tight_layout()
#     plt.imshow(I_generated, cmap='gray')
#     plt.savefig('img/I_generated_{}.png'.format(args.latent))
#     plt.close(fig)

def reconstruction_var_change(path, out, d, l, name, change, i):
    model.load_state_dict(torch.load(out))
    model.eval()
    with torch.no_grad():
        data = d.to(device)
        l = l.to(device)
        u, _, _ = model.var_change(d, l, change, i)
    x_sample = np.squeeze(d.detach().cpu().numpy())
    # print(x_sample.shape)
    I_org = x_sample.reshape(H, W)

    # scipy.io.savemat('mat/15-1200-0:2/changef2/Horg_{}_{}_lin_generated_lin_{}_{}_{}.mat'.format(i, change, args.latent, args.epoch, k),
    #                  mdict={'H_recon_org': I_org})
    # fig = plt.figure()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.imshow(I_org, cmap='gray')
    # plt.savefig('image/Horg_{}_{}_lin_generated_lin_{}_{}_{}.png'.format(i, change, args.latent, args.epoch, k))
    # plt.close(fig)
    u = np.squeeze(u.detach().cpu().numpy())
    I_reconstructed = u.reshape(H, W)
    # print(I_reconstructed)
    scipy.io.savemat(path + 'traversing/Hchange_H_{}_{}_{}_CNNcvae_{}_{}.mat'.format(name, i, change, args.latent, args.epoch),
        mdict={'H_recon': I_reconstructed})
    # fig = plt.figure()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.imshow(I_reconstructed, cmap='gray')
    # plt.savefig('image/15-1200-0:2/Hchange{}__{}_lin_generated{}_{}_{}.png'.format(i, change, args.latent, args.epoch, k))
    # plt.close(fig)


def reconstruction(path, out, d , l,  names, name, modname):
    model.load_state_dict(torch.load(out))
    model.eval()
    with torch.no_grad():
        data = d.to(device)
        # print("recon")
        # print(data.size())
        # data = np.zeros((16, 120, 1786))
        # data = torch.from_numpy(data).float().cuda()
        l = l.to(device)
        # print("label" + str(l.shape))
        # u, mu, logvar, mu2, logvar2
        recon_data, _, _ , _, _, names = model(data, l, names)
        # print(str(recon_data.shape) + "recon shape recon")
    # print(recon_data.size())
    recon_data = recon_data[name]
    # print(recon_data)
    data = data[name]
    data_name = names[name]
    # print(data_name)
    d = d[name]
    x_sample = np.squeeze(data.detach().cpu().numpy())
    I_org = x_sample.reshape(H, W)
    I = np.ones((120, 1786))
    I_org = I_org*(xmax -xmin) + I*xmin
    scipy.io.savemat(path + 'recon/' + modname +str(name) + ' H_org_{}_{}.mat'.format(args.latent, args.epoch),
                     mdict={'H_org': I_org})
    # fig = plt.figure()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.imshow(I_org, cmap='gray')
    # plt.savefig('CNN_results/Mat/recon/' + name + '_' + 'H_org_{}_{}.png'.format(args.latent, args.epoch))
    # plt.close(fig)
    x_reconstruct = np.squeeze(recon_data.detach().cpu().numpy())
    I_reconstructed = x_reconstruct.reshape(H, W)
    # I_reconstructed = x_reconstruct.reshape(H, W)
    I = np.ones((120, 1786))
    # print(I_reconstructed)
    I_reconstructed = I_reconstructed*(xmax -xmin) + I*xmin
    # print(I_reconstructed)
    scipy.io.savemat(path + 'recon/' + modname +  str(name)  + 'H_recon_{}_{}.mat'.format(args.latent, args.epoch),
                     mdict={'H_recon': I_reconstructed})
    # fig = plt.figure()
    # plt.axis('off')
    # plt.tight_layout()
    # plt.imshow(I_reconstructed, cmap='gray')
    # plt.savefig('CNN_results/Mat/recon/' + name + '_' + 'H_recon_{}_{}.png'.format(args.latent, args.epoch))
    # plt.close(fig)
    # print("sizeeeeeees")
    # print(recon_data.size())
    # print(d.size())
    MSE_normal = F.mse_loss(recon_data, data)
    print(data_name)
    print("normal recon loss with org =" + str(MSE_normal.item()))



def generate(path, out, modname, labs, names):
    model.load_state_dict(torch.load(out))
    # model.eval()
    # model = torch.load(out)
    # model.load_state_dict(model)
    # print(model)
    model.eval()
    # print(model)
    generated_data, names = model.generate_lab_(labs, names)
    generated_data = np.squeeze(generated_data.detach().cpu().numpy())

    for i in range(4):
        I_gen = generated_data[i].reshape(H, W)
        scipy.io.savemat(path + 'gen/' + modname +'_' + 'H_generated_H0_{}_{}_{}.mat'.format(args.latent, args.epoch, i),
                         mdict={'H_gen': I_gen})
        print("generate" + names[i])

def generate_only_geo(path, out, modname, labs, names):
    model.load_state_dict(torch.load(out))
    # model.eval()
    # model = torch.load(out)
    # model.load_state_dict(model)
    # print(model)
    model.eval()
    # print(model)
    generated_data, names = model.generate_onlygeo(labs, names)
    generated_data = np.squeeze(generated_data.detach().cpu().numpy())

    for i in range(4):
        I_gen = generated_data[i].reshape(H, W)
        scipy.io.savemat(path + 'gen/' + modname +'_' + 'H_generated_onlygeo_{}_{}_{}.mat'.format(args.latent, args.epoch, i),
                         mdict={'H_gen': I_gen})
        print("generate" + names[i])

def generate_onegeo(path, out, modname, labs, names, num, flag):
    model.load_state_dict(torch.load(out))
    model.eval()
    # print(model)
    generated_data, names = model.generate_onegeo(labs, names, flag)
    generated_data = np.squeeze(generated_data.detach().cpu().numpy())
    I_gen = generated_data[0].reshape(H, W)
    scipy.io.savemat(path + 'gen_geo/' + modname +'_' + 'H_generated_H0_{}_{}_{}.mat'.format(args.latent, args.epoch, num),
                     mdict={'H_gen': I_gen})
    print("generate" + names[0])


def transformation():
    model.eval()
    test_loader_tr = DataLoader(test_dataset, batch_size=3000, shuffle=False)
    with torch.no_grad():
        data, target = next(iter(test_loader_tr))
        data = data.to(device)
        mu, logvar = model.encode(data)
        z = model.reparameterize(mu, logvar)
    z = z.data
    z = np.squeeze(z.detach().cpu().numpy())
    target = target.detach().cpu().numpy()
    fig = plt.figure()
    plt.figure(figsize=(10, 8)) 
    plt.scatter(z[:, 0], z[:, 1], c=target, s=20)
    plt.colorbar()
    plt.grid()
    plt.savefig('img/I_transform.png')
    plt.close(fig)

def diagnose(path, out, d , l, names, modname):
    model.load_state_dict(torch.load(out))
    model.eval()
    with torch.no_grad():
        datas = d.to(device)
        l = l.to(device)
        recon_datas, _, _ , _, _, names = model(datas, l, names)
        # print(str(recon_data.shape) + "recon shape recon")
    # print(recon_data.size())
    for i in range(10):
        recon_data = recon_datas[i]
        data = datas[i]
        data_name = names[i]
        print(data_name)
        ind = data_name.find("glob")
        real_H_pth = '/home/mt6129/H/final-data/real-H_processed/' + 'H_' + data_name[ind:]
        # label = file[g_ind:]
        matFile = scipy.io.loadmat(real_H_pth)
        real_H = matFile['h']
        scipy.io.savemat(path + 'diagnose/' + modname + str(i) + '_' + ' H_real_{}_{}.mat'.format(args.latent, args.epoch),
                        mdict={'H_real': real_H})
        # d = d[i]
        x_sample = np.squeeze(data.detach().cpu().numpy())
        I_org = x_sample.reshape(H, W)
        I = np.ones((120, 1786))
        I_org = I_org*(xmax -xmin) + I*xmin
        scipy.io.savemat(path + 'diagnose/' + modname + str(i) + '_' + ' H_data_{}_{}.mat'.format(args.latent, args.epoch),
                        mdict={'H_org': I_org})
        # fig = plt.figure()
        # plt.axis('off')
        # plt.tight_layout()
        # plt.imshow(I_org, cmap='gray')
        # plt.savefig('CNN_results/Mat/recon/' + name + '_' + 'H_org_{}_{}.png'.format(args.latent, args.epoch))
        # plt.close(fig)
        x_reconstruct = np.squeeze(recon_data.detach().cpu().numpy())
        I_reconstructed = x_reconstruct.reshape(H, W)
        # I_reconstructed = x_reconstruct.reshape(H, W)
        I = np.ones((120, 1786))
        # print(I_reconstructed)
        I_reconstructed = I_reconstructed*(xmax -xmin) + I*xmin
        # print(I_reconstructed)
        scipy.io.savemat(path + 'diagnose/' + modname + str(i) + '_' +  'H_recon_{}_{}.mat'.format(args.latent, args.epoch),
                        mdict={'H_recon': I_reconstructed})
        # fig = plt.figure()
        # plt.axis('off')
        # plt.tight_layout()
        # plt.imshow(I_reconstructed, cmap='gray')
        # plt.savefig('CNN_results/Mat/recon/' + name + '_' + 'H_recon_{}_{}.png'.format(args.latent, args.epoch))
        # plt.close(fig)
        # print("sizeeeeeees")
        # print(recon_data.size())
        # print(d.size())
        # print(type(recon_data))
        # print(type(data))
        MSE_normal = F.mse_loss(recon_data, data)
        # print(data_name)
        print("normal recon loss with data =" + str(MSE_normal.item()))
        MSE_normal_real = F.mse_loss(recon_data, torch.from_numpy(real_H).float().to(device))
        print("normal recon loss with real H =" + str(MSE_normal_real.item()))
        print("--------------------------------------------------------------")

def latent():
    n = 20
    x = np.linspace(-2, 2, n)
    y = np.linspace(-2, 2, n)

    I_latent = np.empty((H*n, W*n))
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            z = np.array([[xi, yi]]*BATCH_SIZE)
            with torch.no_grad():
                z = torch.tensor(z, device=device).float()
                x_hat = model.decode(z)
            x_hat = x_hat.data
            x_hat = np.squeeze(x_hat.detach().cpu().numpy())
            I_latent[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = x_hat[0].reshape(28, 28)

    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(I_latent, cmap="gray")
    plt.savefig('img/I_latent.png')
    plt.close(fig)


# # ##################3 uncomment for opt #################
# out = '/home/mt6129/H/model/model_CNN_CVAE_move_8_900'
# model.load_state_dict(torch.load(out))
# ymat = scipy.io.loadmat('/home/mt6129/H/inf-dense/BSP_1786.mat')
org_H = scipy.io.loadmat('/home/mt6129/H/final-data/recon/H_ax_3_-20_globswiy20swix10zrot0.mat')
geo =  scipy.io.loadmat('/home/mt6129/H/final-data/recon_geo/H_ax_3_-20_globswiy20swix10zrot0.mat')



root_path = '/home/mt6129/H/final-data/adapting/BSP_ax3-20_swiy20swix10zrot0/'
files = os.listdir(root_path)
tot_data = len(files)
print(tot_data)
ys = np.zeros((tot_data, 120, 201))
names = np.empty(tot_data, dtype=object)
num = 0
for i, file in enumerate(files):
    print(file)
    names[i] = file[4:]
    # names[i] = names[i][29,:]
    print(names[i])
    path = os.path.join(root_path, file)
    matFile = scipy.io.loadmat(path)
    y = matFile['bsp']
    # resize u
    # u = cv2.resize(u, (478, 118), interpolation=cv2.INTER_LINEAR)
    # normalized = (y-y.min())/(y.max()-y.min())
    ys[i, :, :] = y
    # dataset[i, :, :] = x
    #############
    num = num+1

# exit()
# y = ymat['bsp']
# normalized = (y-y.min())/(y.max()-y.min())
# ys[tot_data, :, :] = normalized

# quit()
org_H = org_H['h']
real_H = org_H
# # print("blalalala")
g = geo['geo']
# print(y.size)
# # quit()lam*


def minim_func(z, x, y, zg, lam):
    # print("here")
    # f = LA.norm(np.squeeze(model.decode(torch.from_numpy(z).to(device).reshape((1, 8, 5, 12)).float().cuda()).detach().cpu().numpy()).dot(x) - y) + LA.norm(x) + LA.norm(z)
    f = LA.norm(np.squeeze(model.decode(torch.cat((torch.from_numpy(z).to(device).reshape((1, 8, 5, 12)).float().cuda(), zg) , dim=1 )).detach().cpu().numpy()*(xmax -xmin) + xmin).dot(x) - y) + LA.norm(x)
    return f

    # ||h(z)x - y|| + ||x|| + ||z|| + ||h(z)x - y2|| + ||h(z)x - y3||
def opt():
    # x = np.ones((1786, 199))
    #load model
    m = 'model_CNN_CVAE_rots_strong_decoder_newencoder_8_2200_b1_0.001_b2_0'
    out='model/' + m 
    model.load_state_dict(torch.load(out))

    #get the groundtruth data
    dataloader_iterator = iter(recon_loader)
    data, lab, names_ = next(iter(dataloader_iterator))
    # print(size(d)
    # data = d[0]
    # print(d)
    # lab = d[1]
    # print(lab)
    # quit()
    print("shape" + str(data.shape))

    # reconstruction(out='model/' + m, d=data, l = lab, name=0, modname = m)
    data = data.to(device)
    # org_H = data
    mu, logvar = model.encode(data.reshape(1, 1, 120,1786))
    z = model.reparameterize(mu, logvar)
    lab = lab.to(device)
    mu2, logvar2 = model.encode2(lab.reshape(1, 1, 120,1786))
    z2 = model.reparameterize(mu2, logvar2)
    z = torch.cat((z, z2),  dim=1)
    Hz = model.decode(z)*(xmax -xmin) + xmin
    Hrec = np.squeeze(Hz.detach().cpu().numpy()).reshape(120, 1786)
    scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/Hrec.mat', {'Hrec' : Hrec})
    print("here at leaaaaaaast")
    # z = z.reshape((8, 5, 12))
    # # print(z.size())
    # z = np.squeeze(z.detach().cpu().numpy())
    # z = z.flatten()
    
    # print("z isze" + str(z.size))
    g = lab
    z , Hz, zr, zg = model.generate_lab(g)
    # z = z[0]
    # Hz = Hz[0]
    # exit()
    Hz = Hz *(xmax -xmin) + xmin
    initH = np.squeeze(Hz.detach().cpu().numpy()).reshape(120, 1786)
    scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/initH.mat', {'initH' : initH})
    # z = z.reshape((16, 5, 12))
    # print(z.size())
    z = np.squeeze(zr.detach().cpu().numpy())
    z = z.flatten()
    # print(z.shape)
    I = np.identity(1786)
    update = 1
    z_comp = z
    loopnum = 50
    xs = np.ones((tot_data, 1786, 201))
    lam = 0.1
    for i in range(1):
        # print(names[i])
        xs[i , :, :] = (LA.inv(initH.transpose().dot(initH) + lam * (I.transpose().dot(I)))).dot(initH.transpose().dot(ys[i , :, :])) 
        scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/initx_' + '_' + names[i], {'initx' : xs[i]})
        init_frwsol = initH.dot(xs[i])    
        scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/init_frwsol_' + names[i], {'init_frwsol' : init_frwsol})
        
    # x = (LA.inv(initH.transpose().dot(initH) + 0.0001 * (I.transpose().dot(I)))).dot(initH.transpose().dot(y)) 
    # x1 = x
    # x2 = x
    # x3 = x
    # x4 = x
    # scipy.io.savemat('/home/mt6129/H/opt_answers/cvae/initx1786_Seg1exc1_test.mat', {'initx' : x})
    lower = -5 * np.ones(480)
    upper = 5*np.ones(480)
    # print(lower)
    import logging
    for j in range(loopnum):
        print("enter main loop at iter" + str(j))
        if j < 15:
            lam = 0.7 * lam
        for i in range(1):
            y = ys[i]
            # print(y.shape)
            x = xs[i]
            # print(x.shape)
            logging.basicConfig(level=logging.INFO, format='%(message)s')
            # print("before bob")
            # do_logging=True, print_progress=True,
            soln = pybobyqa.solve(minim_func, z, args = (x, y, zg, lam), print_progress=True, bounds = (lower , upper), objfun_has_noise=True, maxfun = 5000, seek_global_minimum=True, rhobeg = 5, rhoend = 0.00001)
            z = soln.x
            z = torch.from_numpy(z).to(device).reshape((1, 8, 5, 12))
            z = z.float().cuda() 
            z_h = torch.cat((z, zg),  dim=1)
            H = model.call_dec(z_h) *(xmax -xmin) + xmin
            H = np.squeeze(H.detach().cpu().numpy())
            H = H.reshape(120, 1786)
            Hz = H
            xs[i , :, :] = (LA.inv(H.transpose().dot(H) + lam * (I.transpose().dot(I)))).dot(H.transpose().dot(ys[i , :, :])) 
            # if(j != loopnum - 1):
            z = z.detach().cpu().numpy()
            z = z.astype('float64')
            z = z.flatten()
        # if (z_comp == z):
        #     break
        # z_comp = z
        # if(i != loopnum - 1):
        #     z = z.detach().cpu().numpy()
        #     z = z.astype('float64')
        #     z = z.flatten()
    frwsol = np.ones((20, 120, 201))
    for i in range(1):       
        frwsol = Hz.dot(xs[i])    
        x = xs[i]
        scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/frwsol_' +'_' + names[i], {'Hx' : frwsol})
        scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/x_' + '_' + names[i], {'x' : x})
        bsl_x = (LA.inv(org_H.transpose().dot(org_H) + 0.000001 * (I.transpose().dot(I)))).dot(org_H.transpose().dot(ys[i]))    
        scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/bsl_x_' + names[i], {'bsl_x' : bsl_x})
        bsl_frwsol = Hz.dot(xs[i])    
        scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/bsl_frwsol_.mat' + names[i], {'bsl_frwsol' : bsl_frwsol})

        
    
    # bsp = ys[tot_data]
    # print(soln.message)
    return z, Hz

def main():
    # train_vae()
    dataloader_iterator = iter(test_loader)
    # # # # #best model
    m = 'model_CNN_CVAE_rots_strong_decoder_newencoder_8_2200_b1_0.001_b2_0'

    # #the final latent error was around 30 and again not ,uch changes in H traversing. Only the last feature had some changes in small peaks
    # #1) try the strong decoder again and do the annealing and try to get a bigger latent error
    # # m = 'model_CNN_CVAE_rots_final_weakdecodermove_8_500_b1_1e-05_b2_0'
    # #check below
    # # m = 'model_CNN_CVAE_rots_finalmove_8_201_b1_0.001_b2_0.001'
    # # m = 'model_CNN_CVAE_all_normalizedmove_geoz2_8_250'
    z, H = opt()
    scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/z_allseg.mat', {'z' : z})
    # scipy.io.savemat('/home/mt6129/H/opt_answers/cvae/x1786_Seg1exc1.mat', {'x' : x})
    scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/H_allseg.mat', {'H' : H})
    # scipy.io.savemat('/home/mt6129/H/opt_answers/cvae/frwsol1786_Seg1exc1.mat', {'Hx' : frwsol})
    # scipy.io.savemat('/home/mt6129/H/opt_answers/cvae/bsp1786_Seg1exc1.mat', {'y' : y})
    # scipy.io.savemat('/home/mt6129/H/opt_answers/cvae-2channel-multisig-50-1000/bsl_x1786_Seg1exc1.mat', {'bsl_x' : bsl_x})
    # scipy.io.savemat('/home/mt6129/H/opt_answers/cvae-2channel-multisig-50-1000/bsl_frw1786_Seg1exc1.mat', {'bsl_frw' : bsl_frw})
    # scipy.io.savemat('/home/mt6129/H/opt_answers/dense/initH1786_Seg1exc1.mat', {'initH' : initH})
    scipy.io.savemat('/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/orgH.mat', {'real_H' : real_H})

    # uncomment for testing the model
    # for i in range(7):     
    #     data, labs, names = next(dataloader_iterator)

    # path = 'CNN_results/new_enc_8_2200_0.001/'
    # # diagnose(path, out = 'model/' + m, d = data, l = labs, names = names, modname = m)
    # # # # # data = next(iter(train_loader))
    # data = data.to(device)
    # labs = labs.to(device)
    # # # data1 = data[0]
    # # # lab1 = labs[0]
    # # # data2 = data[1]
    # # # lab2 = labs[1]
    # data3 = data[0]
    # lab3 = labs[0]
    # # lab4= labs[3]
    # # # # # #m is the number
    # generate_only_geo(path, out='model/' + m , modname = m, labs = labs, names = names)

    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=0, modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=1, modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=2, modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=3 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=4 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=5 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=6 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=7 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=8 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=9 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=10 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=11 , modname = m)
    # reconstruction(path, out='model/' + m, d=data, l=labs, names = names, name=12 , modname = m)
    
    # generate(path, out='model/' + m , modname = m, labs = labs, names = names)
    # generate_onegeo(path, out='model/' + m , modname = m, labs = labs, names = names, num = 0, flag = 0)
    # generate_onegeo(path, out='model/' + m , modname = m, labs = labs, names = names, num = 1, flag = 0)
    # generate_onegeo(path, out='model/' + m , modname = m, labs = labs, names = names, num = 2, flag = 0)
    # generate_onegeo(path, out='model/' + m , modname = m, labs = labs, names = names, num = 3, flag = 1)
    

    # # # # # # #on data1 1 no 2 yes 3 no 4no 5yes 6maybe 7maybe 8maybe 9no 10yes 11yes 12mostly no 13maybe 14yes 15maybe 16
    # for j in range(8):
    #     val = j
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=-5, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=-4, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=-3, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=-2, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=-1, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=0, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=1, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=2, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=3, i = val)
    #     reconstruction_var_change(path, out='model/' + m  , d=data3, l=lab3, name='data3', change=4, i = val)
    


if __name__ == "__main__":
    main()