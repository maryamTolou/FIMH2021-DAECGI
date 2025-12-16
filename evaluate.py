"""
In this script, we want to evaluate the already trained model.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io
# import csv
# import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.image as img
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--latent', type=int, default=16, help='latent dimension')
parser.add_argument('--epoch', type=int, default=1000, help='epoch for VAE')
parser.add_argument('--pre_train', type=int, default=0, help='0 for new model, 1 for pre-train')
args = parser.parse_args()

# Hyperparameters
BATCH_SIZE = 4
W = 477
H = 120
# lr = 1e-4
lr = 1e-3

# Prepare dataset
root_path = '/home/mt6129/hearts/Data/Processed_Transepi/'
files = os.listdir(root_path)
tot_data = len(files)

dataset_org = np.zeros((tot_data, H, W))
for i, file in enumerate(files):
    path = os.path.join(root_path, file)
    matFile = scipy.io.loadmat(path)
    u = matFile['h']
    # resize u
    # u = cv2.resize(u, (478, 118), interpolation=cv2.INTER_LINEAR)
    dataset_org[i, :, :] = u
    # print(len(dataset[i,:,:]))


# print(np.min(dataset_org))
# print(np.max(dataset_org))
#normalizing the dataset between 0 and 1
# dataset = (dataset_org - np.min(dataset_org))/np.ptp(dataset_org)
dataset = ((dataset_org - np.min(dataset_org)) * (1/np.ptp(dataset_org) - np.min(dataset_org)) * 255).astype('uint8')
print(np.min(dataset))
print(np.max(dataset))
# print(dataset)
recon_test_data = scipy.io.loadmat('/home/mt6129/hearts/Data/Processed_Transepi/Trans_epi_s-5v0.mat')
u_rec = matFile['h']

np.random.seed(45)
rnd_idx = np.random.permutation(tot_data)

train_split, test_split = rnd_idx[:int(0.8 * tot_data)], rnd_idx[int(0.8 * tot_data):]
data_list_train = [dataset[i] for i in train_split]
data_list_test = [dataset[i] for i in test_split]
data_list_train = np.array(data_list_train)
data_list_test = np.array(data_list_test)
recon_list = np.zeros((1, H, W))
recon_list[0, :, :] = u_rec
recon_set = np.array(recon_list)

class CustDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx, :]
        return x


train_dataset = CustDataset(data_list_train)
# print(train_dataset[1])
test_dataset = CustDataset(data_list_test)
recon_dataset = CustDataset(recon_set)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
plot_train_loader = train_loader
print("train_loader shape" + str(len(train_loader)))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
recon_loader = DataLoader(recon_dataset)
print("len recon loader" + str(next(iter(recon_loader)).shape))
# print(train_dataset.__getitem__(1))


# Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, latent_dim):
        # architecture 1: smooth not good results
        # super().__init__()
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=(1, 2), padding=(1, 0))
        # # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=(1, 2), padding=(1, 1))
        # # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=(0,1))
        # # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #
        # self.fc1 = nn.Conv2d(32, 32, kernel_size=1)
        # self.fc21 = nn.Conv2d(32, latent_dim, kernel_size=1)
        # self.fc22 = nn.Conv2d(32, latent_dim, kernel_size=1)
        #
        # self.fcd3 = nn.Conv2d(latent_dim, 32, kernel_size=1)
        # self.fcd4 = nn.Conv2d(32, 32, kernel_size=1)
        #
        # # self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3)
        # # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=(1, 2), padding=(1, 1))
        # # self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dconv1 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=(1, 2), padding=(1, 1))

        # # architecture 2:
        # super().__init__()
        # # self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=(1, 1), padding=(0, 0))
        # self.fc1 = nn.Conv2d(1, latent_dim, kernel_size=(120,477))
        # self.fcd1 = nn.ConvTranspose2d(latent_dim, 1, kernel_size=(120,477))
        # # self.fc12 = nn.Conv2d(8, latent_dim, kernel_size=1)

        # # architecture 3:
        # super().__init__()
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=(1, 2), padding=(1, 0))
        # # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=(1, 2), padding=(1, 1))
        # # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=3, padding=(0, 1))
        # # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        #
        # self.fc21 = nn.Conv2d(32, latent_dim, kernel_size=1)
        # self.fc22 = nn.Conv2d(32, latent_dim, kernel_size=1)
        #
        # self.fcd3 = nn.Conv2d(latent_dim, 32, kernel_size=1)
        #
        # # self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3)
        # # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=(1, 2), padding=(1, 1))
        # # self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dconv1 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=(1, 2), padding=(1, 1))

        # architecture 4:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=(2, 2), padding=(0, 1))
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=(2, 2), padding=(0, 1))
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=(0, 1))
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.fc21 = nn.Conv2d(32, latent_dim, kernel_size=1)
        self.fc22 = nn.Conv2d(32, latent_dim, kernel_size=1, padding=(1, 0))
        self.fcd3 = nn.Conv2d(latent_dim, 32, kernel_size=1, padding=(1, 1))

        # self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dconv3 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 2),stride=(2, 2), padding=(2, 0))
        # self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dconv2 = nn.ConvTranspose2d(16, 8, kernel_size=(5, 2), stride=(2, 2), padding=(2, 1))
        # self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dconv1 = nn.ConvTranspose2d(8, 1, kernel_size=(7, 2), stride=(2, 2), padding=(1, 2))

        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, data):
        #arch 1
        # x = F.elu(self.conv1(data))
        # # x, self.ind1 = self.pool1(x)
        # x = F.elu(self.conv2(x))
        # # x, self.ind2 = self.pool2(x)
        # x = F.elu(self.conv3(x))
        # # x, self.ind3 = self.pool3(x)
        # x = self.fc1(x)
        # mu = self.fc21(x)
        # logvar = self.fc22(x)

        # arch2:
        # x = F.elu(self.fc1(data))
        # mu = self.fc1(data)
        # logvar = self.fc1(data)


        #arch3
        x = F.elu(self.conv1(data))
        # x, self.ind1 = self.pool1(x)
        x = F.elu(self.conv2(x))
        # x, self.ind2 = self.pool2(x)
        x = F.elu(self.conv3(x))
        # x, self.ind3 = self.pool3(x)
        mu = self.fc21(x)
        logvar = self.fc21(x)
        mu = self.fc22(x)
        logvar = self.fc22(x)
        return mu, logvar

    def decode(self, z):
        # print("shape of z")
        # print(z.shape)
        # x = self.fcd3(z)
        # x = self.fcd4(x)
        # # x = self.unpool3(x, indices=self.ind3)
        # x = F.elu(self.dconv3(x))
        # # x = self.unpool2(x, indices=self.ind2)
        # x = F.elu(self.dconv2(x))
        # # x = self.unpool1(x, indices=self.ind1)
        # x = torch.sigmoid(self.dconv1(x)) #sigmoid here if normalization can be done else linear layer

        #arch2
        # x =  self.fcd1(z)

        #arch3
        print(" in decodeee")
        print(z.shape)
        x = self.fcd3(z)
        print(x.shape)
        x = F.elu(self.dconv3(x))
        print(x.shape)
        # x = self.unpool2(x, indices=self.ind2)
        x = F.elu(self.dconv2(x))
        print(x.shape)
        # x = self.unpool1(x, indices=self.ind1)
        x = self.dconv1(x)  # sigmoid here if normalization can be done else linear layer
        print(x.shape)
        return x

    def generate(self, n = 5):
        temp_ = torch.randn(n, self.latent_dim, 40, 40)
        z = torch.rand_like(temp_).to(device)
        x_gen = self.decode(z)
        return x_gen

    def forward(self, data):
        N, h, w = data.shape
        data = data.view(N, -1, h, w)
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        # print("here z dim" + str(z.shape))
        u = self.decode(z)
        u = u.view(N, h, w)
        return u, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD, BCE + KLD


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
    model_state = torch.load(os.path.join(model_dir, 'm_latest_{}_{}'.format(args.latent, args.epoch)),
                             map_location='cuda:0')
    model.load_state_dict(model_state)


def inline_print(s):
    sys.stdout.write(s + '\r')
    sys.stdout.flush()

def train(epoch):
    # cwd = os.getcwd()
    # print(cwd)
    model.train()
    recon, latent, total = 0, 0, 0
    n = 0
    for x in train_loader:


        x = x.to(device)
        # print(x.shape)

        optimizer.zero_grad()
        u, mu, logvar = model(x)
        recon_loss, latent_loss, total_loss = loss_function(u, x, mu, logvar)

        total_loss.backward()
        recon += recon_loss.item()
        latent += latent_loss.item()
        total += total_loss.item()
        optimizer.step()

        n += 1
        log = '[Epoch {:03d}] Loss: {:.4f}, Recon loss: {:.4f}, Latent loss: {:.4f}'.format(epoch,
                                                                                            total / (n * BATCH_SIZE),
                                                                                            recon / (n * BATCH_SIZE),
                                                                                            latent / (n * BATCH_SIZE))
        inline_print(log)

    torch.save(model.state_dict(), model_dir + '/m_latest_{}_{}'.format(args.latent, args.epoch))
    # print(cwd)
    return recon, latent, total


def test():
    model.eval()
    recon, latent, total = 0, 0, 0
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)

            u, mu, logvar = model(x)
            recon_loss, latent_loss, total_loss = loss_function(u, x, mu, logvar)

            recon += recon_loss.item()
            latent += latent_loss.item()
            total += total_loss.item()
    return recon, latent, total


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
    for epoch in range(args.epoch):
        recon_train, latent_train, total_train = train(epoch)
        recon_test, latent_test, total_test = test()

        recon_train /= len(train_dataset)
        latent_train /= len(train_dataset)
        total_train /= len(train_dataset)
        recon_test /= len(test_dataset)
        latent_test /= len(test_dataset)
        total_test /= len(test_dataset)
        log = '[Epoch {:03d}] Loss: {:.4f}, Recon loss: {:.4f}, Latent loss: {:.4f}'.format(epoch, total_test,
                                                                                            recon_test, latent_test)
        print(log)
        recon_train_a.append(recon_train)
        latent_train_a.append(latent_train)
        total_train_a.append(total_train)
        recon_test_a.append(recon_test)
        latent_test_a.append(latent_test)
        total_test_a.append(total_test)
    plot_losses(recon_train_a, recon_test_a, 'reconstruction', args.epoch)
    plot_losses(latent_train_a, latent_test_a, 'latent', args.epoch)
    plot_losses(total_train_a, total_test_a, 'total', args.epoch)


def reconstruction():
    model.eval()
    with torch.no_grad():
        data = next(iter(test_loader))
        print(data.shape)
        # print(type(data))
        # data = next(iter(recon_loader))
        # print(str(data.shape)+ "shape recon")
        data = data.to(device)

        # x_recon_org = data
        recon_data,_, _ = model(data)
        print(str(recon_data.shape)+ "recon shape recon")


    # print(str(len(recon_data)) + ' hahaha')

    # import ipdb
    # ipdb.set_trace()
    # x_sample = data.view(BATCH_SIZE, -1)
    x_sample = np.squeeze(data.detach().cpu().numpy())
    # print(x_sample.shape)
    I_org = x_sample[0].reshape(H, W)
    scipy.io.savemat('H_recon_org.mat', mdict={'H_recon_org': I_org})
    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(I_org, cmap='gray')
    plt.savefig('img/I_org_xia_{}.png'.format(args.latent))
    plt.close(fig)
    # print(x_sample.shape)
    # x_reconstruct = recon_data.view(BATCH_SIZE, -1)
    x_reconstruct = np.squeeze(recon_data.detach().cpu().numpy())
    # I_reconstructed = np.empty((H, W))
    I_reconstructed = x_reconstruct[0].reshape(H, W)
    # n = np.sqrt(BATCH_SIZE).astype(np.int32)
    # I_reconstructed = np.empty((H * n, 2 * W * n))

    # for i in range(n):
    #     for j in range(n):
    #         x = np.concatenate(
    #             (x_reconstruct[i * n + j, :].reshape(H, W),
    #              x_sample[i * n + j, :].reshape(H, W)),
    #             axis=1
    #         )
    #         I_reconstructed[i * H:(i + 1) * H, j * 2 * W:(j + 1) * 2 * W] = x

    # x = np.concatenate(
    #     (x_reconstruct[0, :].reshape(H, W),
    #     x_sample[0, :].reshape(H, W)),
    #     axis=1
    #     )

    # I_reconstructed = x_reconstruct[0].reshape(H, W)

    scipy.io.savemat('H_recon_lin.mat', mdict={'H_recon': I_reconstructed})
    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(I_reconstructed, cmap='gray')
    plt.savefig('img/lin_reconstructed_xia_{}.png'.format(args.latent))
    plt.close(fig)



def generate(out):

    model.load_state_dict(torch.load(out))
    model.eval()

    generated_data = model.generate(n=3)
    generated_data = np.squeeze(generated_data.detach().cpu().numpy())


    for i in range(3):
        I_reconstructed = generated_data[i].reshape(H, W)
        scipy.io.savemat('H_generated_{}.mat'.format(i), mdict={'H_recon': I_reconstructed})
        fig = plt.figure()
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(I_reconstructed, cmap='gray')
        plt.savefig('img/generated{}_{}.png'.format(args.latent, i))
        plt.close(fig)


def plot_H():
    for x in train_loader:
        x = x.to(device)
    x_plot = x
    x_plot = np.squeeze(x_plot.detach().cpu().numpy())
    # n = np.sqrt(BATCH_SIZE).astype(np.int32)
    I_plot = np.empty((H , W))
    I_plot = x_plot[0, :].reshape(H, W)
    # print(str(I_plot.shape) + "len iplor")
    fig = plt.figure()
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(I_plot, cmap='gray')
    plt.savefig('img/H_plot_org{}.png'.format(args.latent))
    plt.close(fig)


def main():
    #generate
    train_vae()
    reconstruction()
    generate(out='model/m_latest_16_1000')


if __name__ == "__main__":
    main()
