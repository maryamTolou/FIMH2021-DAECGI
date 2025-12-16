"""
Cleaned-up single-file script (same behavior, less clutter).

Notes on what I *didn't* change (to avoid breaking runs):
- Kept your model architecture + tensor shapes as-is.
- Kept dataset normalization using global xmin/xmax and gmin/gmax.
- Kept your optimization loop structure and file outputs.
- Left hard-coded paths (but grouped them + made them easy to change).
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from numpy import linalg as LA
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pybobyqa
import matplotlib.pyplot as plt


# -----------------------------
# Config / CLI
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class Paths:
    H_train: str = "/home/mt6129/H/final-data/train_sub/"
    H_test: str = "/home/mt6129/H/final-data/test_sub/"
    g_train: str = "/home/mt6129/H/final-data/train_geo_sub/"
    g_test: str = "/home/mt6129/H/final-data/test_geo_sub/"
    H_recon: str = "/home/mt6129/H/final-data/recon/"
    g_recon: str = "/home/mt6129/H/final-data/recon_geo/"
    adapting_root: str = "/home/mt6129/H/final-data/adapting/BSP_ax3-20_swiy20swix10zrot0/"
    recon_H_mat: str = "/home/mt6129/H/final-data/recon/H_ax_3_-20_globswiy20swix10zrot0.mat"
    recon_geo_mat: str = "/home/mt6129/H/final-data/recon_geo/H_ax_3_-20_globswiy20swix10zrot0.mat"
    out_root: str = "/home/mt6129/H/opt_answers/new_enc_8_2200_0.001_1_newterm_50it_15lmd_varlambda/"


@dataclass(frozen=True)
class Hyper:
    batch_size: int = 16
    H: int = 120
    W: int = 1786
    lr: float = 1e-4
    b1: float = 0.001
    b2: float = 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--latent", type=int, default=8, help="latent dimension")
    p.add_argument("--epoch", type=int, default=2200, help="epochs for VAE")
    p.add_argument("--pre_train", type=int, default=0, help="0=new, 1=load pretrain")
    return p.parse_args()


def ensure_dirs(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def inline_print(msg: str) -> None:
    sys.stdout.write(msg + "\r")
    sys.stdout.flush()


# -----------------------------
# Data utils
# -----------------------------
def list_sorted(dir_path: str) -> List[str]:
    # Important: stable ordering so H/geo align by index.
    return sorted([f for f in os.listdir(dir_path) if not f.startswith(".")])


def find_min_max(H_root: str, g_root: str) -> Tuple[float, float, float, float]:
    H_files = list_sorted(H_root)
    g_files = list_sorted(g_root)
    if len(H_files) != len(g_files):
        print(f"[WARN] {H_root} and {g_root} have different counts: {len(H_files)} vs {len(g_files)}")

    xmin, xmax = float("inf"), float("-inf")
    gmin, gmax = float("inf"), float("-inf")

    for i, fname in enumerate(H_files):
        if i % 200 == 0:
            print(f"Scanning {H_root}: {i}/{len(H_files)}")

        h_mat = scipy.io.loadmat(os.path.join(H_root, fname))["h"]
        xmin = min(xmin, float(h_mat.min()))
        xmax = max(xmax, float(h_mat.max()))

        # by your original logic: label file name == H file name
        g_mat = scipy.io.loadmat(os.path.join(g_root, fname))["geo"]
        gmin = min(gmin, float(g_mat.min()))
        gmax = max(gmax, float(g_mat.max()))

    return xmin, xmax, gmin, gmax


class CustData(Dataset):
    def __init__(self, H_path: str, g_path: str, xmin: float, xmax: float, gmin: float, gmax: float):
        self.H_path = H_path
        self.g_path = g_path
        self.H_files = list_sorted(H_path)
        self.g_files = list_sorted(g_path)
        self.xmin, self.xmax = xmin, xmax
        self.gmin, self.gmax = gmin, gmax

    def __len__(self) -> int:
        return len(self.H_files)

    def __getitem__(self, idx: int):
        H_name = self.H_files[idx]
        H = scipy.io.loadmat(os.path.join(self.H_path, H_name))["h"]
        Hn = (H - self.xmin) / (self.xmax - self.xmin)
        x = torch.from_numpy(Hn).float()

        # match by file name (your original behavior)
        G_name = H_name if H_name in self.g_files else self.g_files[idx]
        G = scipy.io.loadmat(os.path.join(self.g_path, G_name))["geo"]
        Gn = (G - self.gmin) / (self.gmax - self.gmin)
        y = torch.from_numpy(Gn).float()

        return x, y, H_name


# -----------------------------
# Model
# -----------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Encoder 1 (for H)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 15), stride=(2, 5), padding=(1, 7))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 7), stride=(2, 3), padding=(1, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 5), stride=(2, 2), padding=(1, 0))
        self.conv5 = nn.Conv2d(256, 64, kernel_size=(4, 4), stride=(1, 2), padding=(0, 0))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv7 = nn.Conv2d(32, 16, kernel_size=(1, 1))
        self.conv8 = nn.Conv2d(16, 8, kernel_size=(1, 1))
        self.conv91 = nn.Conv2d(8, latent_dim, kernel_size=1)
        self.conv92 = nn.Conv2d(8, latent_dim, kernel_size=1)

        # Decoder (strong)
        self.dconv9 = nn.ConvTranspose2d(2 * latent_dim, 16, kernel_size=1)
        self.dconv8 = nn.ConvTranspose2d(16, 32, kernel_size=1)
        self.dconv7 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.dconv6 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.dconv62 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.dconv5 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.dconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 3), padding=(1, 2))
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 7), stride=(2, 5), padding=(3, 0))
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(10, 14), stride=(3, 5), padding=(2, 1))
        self.dconv1 = nn.ConvTranspose2d(16, 8, kernel_size=(1, 1), stride=(1, 1), padding=(0, 3))
        self.dconv12 = nn.ConvTranspose2d(8, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        # Encoder 2 (for geo)
        self.conv12 = nn.Conv2d(1, 16, kernel_size=(9, 17), stride=(3, 5), padding=(4, 8))
        self.conv22 = nn.Conv2d(16, 32, kernel_size=(5, 11), stride=(2, 5), padding=(1, 5))
        self.conv32 = nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(2, 3), padding=(1, 2))
        self.conv42 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv52 = nn.Conv2d(128, 64, kernel_size=(1, 1))
        self.conv62 = nn.Conv2d(64, 32, kernel_size=(1, 1))
        self.conv72 = nn.Conv2d(32, 8, kernel_size=(1, 1))
        self.conv921 = nn.Conv2d(8, latent_dim, kernel_size=1)
        self.conv922 = nn.Conv2d(8, latent_dim, kernel_size=1)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, data: torch.Tensor):
        x = F.elu(self.conv1(data))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = F.elu(self.conv6(x))
        x = F.elu(self.conv7(x))
        x = F.elu(self.conv8(x))
        mu = F.elu(self.conv91(x))
        logvar = F.elu(self.conv92(x))
        return mu, logvar

    def encode2(self, data: torch.Tensor):
        x = F.elu(self.conv12(data))
        x = F.elu(self.conv22(x))
        x = F.elu(self.conv32(x))
        x = F.elu(self.conv42(x))
        x = F.elu(self.conv52(x))
        x = F.elu(self.conv62(x))
        x = F.elu(self.conv72(x))
        mu = F.elu(self.conv921(x))
        logvar = F.elu(self.conv922(x))
        return mu, logvar

    def decode(self, z: torch.Tensor):
        x = F.elu(self.dconv9(z))
        x = F.elu(self.dconv8(x))
        x = F.elu(self.dconv7(x))
        x = F.elu(self.dconv6(x))
        x = F.elu(self.dconv62(x))
        x = F.elu(self.dconv5(x))
        x = F.elu(self.dconv4(x))
        x = F.elu(self.dconv3(x))
        x = F.elu(self.dconv2(x))
        x = F.elu(self.dconv1(x))
        x = F.elu(self.dconv12(x))
        return x

    def forward(self, data: torch.Tensor, label: torch.Tensor, names):
        N, h, w = data.shape
        data = data.view(N, 1, h, w)
        label = label.view(N, 1, h, w)

        mu, logvar = self.encode(data)
        mu2, logvar2 = self.encode2(label)

        zr = self.reparameterize(mu, logvar)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2), dim=1)

        u = self.decode(z).view(N, h, w)
        return u, mu, logvar, mu2, logvar2, names

    def generate_lab(self, label: torch.Tensor):
        n = 1
        zr = torch.rand((1, 8, 5, 12), device=label.device)
        label = label.view(1, 1, HYP.H, HYP.W)
        mu, logvar = self.encode2(label)
        z2 = self.reparameterize(mu, logvar)
        z = torch.cat((zr, z2), dim=1)
        u = self.decode(z).view(n, HYP.H, HYP.W)
        return z, u, zr, z2

    def generate_lab_(self, label: torch.Tensor, names):
        n = 16
        zr = torch.randn((n, 8, 5, 12), device=label.device)
        label = label.view(n, 1, HYP.H, HYP.W)
        mu2, logvar2 = self.encode2(label)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2), dim=1)
        u = self.decode(z).view(n, HYP.H, HYP.W)
        return u, names

    def generate_onlygeo(self, label: torch.Tensor, names):
        n = 16
        zr = torch.rand((n, 8, 5, 12), device=label.device)
        label = label.view(n, 1, HYP.H, HYP.W)
        mu, logvar = self.encode2(label)
        z2 = self.reparameterize(mu, logvar)
        z = torch.cat((zr, z2), dim=1)
        u = self.decode(z).view(n, HYP.H, HYP.W)
        return u, names

    def generate_onegeo(self, label: torch.Tensor, names, flag: int):
        n = 16
        zr = torch.randn((n, 8, 5, 12), device=label.device)
        if flag == 1:
            zr = zr + (-5) * torch.ones_like(zr)
        label = label.view(n, 1, HYP.H, HYP.W)
        mu2, logvar2 = self.encode2(label)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2), dim=1)
        u = self.decode(z).view(n, HYP.H, HYP.W)
        return u, names

    def var_change(self, data: torch.Tensor, label: torch.Tensor, change: float, i: int):
        h, w = data.shape
        data = data.view(1, 1, h, w)
        label = label.view(1, 1, h, w)

        mu, logvar = self.encode(data)
        mu2, logvar2 = self.encode2(label)

        ch = torch.full((5, 12), float(change), device=data.device)
        mu[0][i] = ch  # keep your original behavior

        zr = self.reparameterize(mu, logvar)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat((zr, z2), dim=1)
        u = self.decode(z)
        return u, mu, logvar

    def call_dec(self, zh: torch.Tensor):
        return self.decode(zh)


def loss_function(recon_x, x, mu, logvar, mu2, logvar2, b: float, b2: float):
    bce = F.mse_loss(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
    total = bce + b * kld + 0.0 * kld2
    return bce, kld, total, kld2


# -----------------------------
# Train / test
# -----------------------------
def train_one_epoch(model: VAE, loader: DataLoader, optimizer, epoch: int, b: float, b2: float):
    model.train()
    recon = latent = total = latent2 = 0.0
    n = 0

    for x, y, names in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        u, mu, logvar, mu2, logvar2, _ = model(x, y, names)
        r, l1, t, l2 = loss_function(u, x, mu, logvar, mu2, logvar2, b, b2)
        t.backward()
        optimizer.step()

        recon += float(r.item())
        latent += float(l1.item())
        total += float(t.item())
        latent2 += float(l2.item())
        n += 1

        inline_print(
            f"[E {epoch:03d}] Loss: {total/(n*HYP.batch_size):.4f}, "
            f"Rec: {recon/(n*HYP.batch_size):.4f}, "
            f"KL1: {latent/(n*HYP.batch_size):.4f}, "
            f"KL2: {latent2/(n*HYP.batch_size):.4f}"
        )

    print()  # newline after inline_print
    return recon, latent, total, latent2


@torch.no_grad()
def eval_epoch(model: VAE, loader: DataLoader, b: float, b2: float):
    model.eval()
    recon = latent = total = latent2 = 0.0

    for x, y, names in loader:
        x = x.to(device)
        y = y.to(device)
        u, mu, logvar, mu2, logvar2, _ = model(x, y, names)
        r, l1, t, l2 = loss_function(u, x, mu, logvar, mu2, logvar2, b, b2)
        recon += float(r.item())
        latent += float(l1.item())
        total += float(t.item())
        latent2 += float(l2.item())

    return recon, latent, total, latent2


def plot_losses(train_a, test_a, loss_type: str, num_epochs: int, latent_dim: int):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xticks(np.arange(1, num_epochs + 1, step=max(1, num_epochs // 20)))
    ax.set_xlabel("epochs")
    ax.plot(train_a, ls="-", label=f"train {loss_type}")
    ax.plot(test_a, ls="-", label=f"test {loss_type}")
    ax.legend(fontsize=14, frameon=False)
    ax.grid(linestyle="--")
    plt.tight_layout()
    fig.savefig(f"img/{loss_type}_{latent_dim}.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -----------------------------
# Reconstruction / generation helpers (kept compatible)
# -----------------------------
def denorm_H(Hn: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return Hn * (xmax - xmin) + xmin


def reconstruction_var_change(path, out, d, l, name, change, i, xmin, xmax):
    model.load_state_dict(torch.load(out, map_location=device))
    model.eval()
    with torch.no_grad():
        d = d.to(device)
        l = l.to(device)
        u, _, _ = model.var_change(d, l, change, i)

    u = np.squeeze(u.detach().cpu().numpy()).reshape(HYP.H, HYP.W)
    scipy.io.savemat(
        os.path.join(path, "traversing", f"Hchange_H_{name}_{i}_{change}_CNNcvae_{ARGS.latent}_{ARGS.epoch}.mat"),
        mdict={"H_recon": u},
    )


def generate_onegeo(path, out, modname, labs, names, num, flag):
    model.load_state_dict(torch.load(out, map_location=device))
    model.eval()
    with torch.no_grad():
        generated_data, names = model.generate_onegeo(labs.to(device), names, flag)
    generated_data = np.squeeze(generated_data.detach().cpu().numpy())
    I_gen = generated_data[0].reshape(HYP.H, HYP.W)
    scipy.io.savemat(
        os.path.join(path, "gen_geo", f"{modname}_H_generated_H0_{ARGS.latent}_{ARGS.epoch}_{num}.mat"),
        mdict={"H_gen": I_gen},
    )
    print("generate", names[0])


# -----------------------------
# Optimization part (your pipeline)
# -----------------------------
def minim_func(z_flat, x, y, zg, lam, xmin, xmax):
    """
    z_flat: (480,) flattened latent for zr of shape (1, 8, 5, 12)
    zg: encoded geo latent already shaped like (1, 8, 5, 12) on GPU
    """
    z = torch.from_numpy(z_flat).to(device).reshape((1, 8, 5, 12)).float()
    z_h = torch.cat((z, zg), dim=1)
    Hz = model.decode(z_h).detach().cpu().numpy()
    Hz = np.squeeze(Hz)  # (120,1786) normalized
    Hz = denorm_H(Hz, xmin, xmax)

    # your objective: ||Hz x - y|| + ||x||
    return LA.norm(Hz.dot(x) - y) + LA.norm(x)


def load_adapting_bsp(root_path: str) -> Tuple[np.ndarray, np.ndarray]:
    files = list_sorted(root_path)
    ys = np.zeros((len(files), 120, 201), dtype=np.float64)
    names = np.empty(len(files), dtype=object)

    for i, file in enumerate(files):
        names[i] = file[4:]
        mat = scipy.io.loadmat(os.path.join(root_path, file))
        ys[i, :, :] = mat["bsp"]

    return ys, names


def opt(paths: Paths, xmin: float, xmax: float):
    # load model
    m = "model_CNN_CVAE_rots_strong_decoder_newencoder_8_2200_b1_0.001_b2_0"
    out = os.path.join("model", m)
    model.load_state_dict(torch.load(out, map_location=device))
    model.eval()

    # recon_loader gives (data, lab, names)
    data, lab, names_ = next(iter(recon_loader))
    data = data.to(device)
    lab = lab.to(device)

    # encode + decode a reconstruction
    mu, logvar = model.encode(data.reshape(1, 1, HYP.H, HYP.W))
    z = model.reparameterize(mu, logvar)
    mu2, logvar2 = model.encode2(lab.reshape(1, 1, HYP.H, HYP.W))
    z2 = model.reparameterize(mu2, logvar2)
    z_full = torch.cat((z, z2), dim=1)

    Hz = model.decode(z_full) * (xmax - xmin) + xmin
    Hrec = np.squeeze(Hz.detach().cpu().numpy()).reshape(HYP.H, HYP.W)
    scipy.io.savemat(os.path.join(paths.out_root, "Hrec.mat"), {"Hrec": Hrec})

    # initialize z from random zr + geo latent
    z_full2, Hz2, zr, zg = model.generate_lab(lab)
    initH = np.squeeze((Hz2 * (xmax - xmin) + xmin).detach().cpu().numpy()).reshape(HYP.H, HYP.W)
    scipy.io.savemat(os.path.join(paths.out_root, "initH.mat"), {"initH": initH})

    z = np.squeeze(zr.detach().cpu().numpy()).astype("float64").flatten()

    ys, seg_names = load_adapting_bsp(paths.adapting_root)
    tot_data = ys.shape[0]

    I = np.identity(HYP.W)
    loopnum = 50
    xs = np.ones((tot_data, HYP.W, 201), dtype=np.float64)
    lam = 0.1

    # init x for first segment only (kept your i range)
    for i in range(1):
        xs[i] = LA.inv(initH.T.dot(initH) + lam * (I.T.dot(I))).dot(initH.T.dot(ys[i]))
        scipy.io.savemat(os.path.join(paths.out_root, f"initx__{seg_names[i]}"), {"initx": xs[i]})
        init_frwsol = initH.dot(xs[i])
        scipy.io.savemat(os.path.join(paths.out_root, f"init_frwsol_{seg_names[i]}"), {"init_frwsol": init_frwsol})

    lower = -5 * np.ones(480)
    upper = 5 * np.ones(480)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    Hz_last = initH
    for j in range(loopnum):
        print(f"enter main loop at iter {j}")
        if j < 15:
            lam = 0.7 * lam

        for i in range(1):
            y = ys[i]
            x = xs[i]

            soln = pybobyqa.solve(
                minim_func,
                z,
                args=(x, y, zg, lam, xmin, xmax),
                print_progress=True,
                bounds=(lower, upper),
                objfun_has_noise=True,
                maxfun=5000,
                seek_global_minimum=True,
                rhobeg=5,
                rhoend=0.00001,
            )
            z = soln.x

            z_t = torch.from_numpy(z).to(device).reshape((1, 8, 5, 12)).float()
            z_h = torch.cat((z_t, zg), dim=1)
            H = model.call_dec(z_h) * (xmax - xmin) + xmin
            H = np.squeeze(H.detach().cpu().numpy()).reshape(HYP.H, HYP.W)
            Hz_last = H

            xs[i] = LA.inv(H.T.dot(H) + lam * (I.T.dot(I))).dot(H.T.dot(y))
            z = z.astype("float64").flatten()

    # save outputs (kept same spirit)
    for i in range(1):
        frwsol = Hz_last.dot(xs[i])
        scipy.io.savemat(os.path.join(paths.out_root, f"frwsol__{seg_names[i]}"), {"Hx": frwsol})
        scipy.io.savemat(os.path.join(paths.out_root, f"x__{seg_names[i]}"), {"x": xs[i]})

    return z, Hz_last


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ARGS = parse_args()
    PATHS = Paths()
    HYP = Hyper()

    ensure_dirs("model", "img")

    # global mins/maxs (train+test)
    xmin1, xmax1, gmin1, gmax1 = find_min_max(PATHS.H_train, PATHS.g_train)
    xmin2, xmax2, gmin2, gmax2 = find_min_max(PATHS.H_test, PATHS.g_test)
    xmin = min(xmin1, xmin2)
    xmax = max(xmax1, xmax2)
    gmin = min(gmin1, gmin2)
    gmax = max(gmax1, gmax2)

    # datasets/loaders
    train_dataset = CustData(PATHS.H_train, PATHS.g_train, xmin, xmax, gmin, gmax)
    test_dataset = CustData(PATHS.H_test, PATHS.g_test, xmin, xmax, gmin, gmax)
    recon_dataset = CustData(PATHS.H_recon, PATHS.g_recon, xmin, xmax, gmin, gmax)

    train_loader = DataLoader(train_dataset, batch_size=HYP.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=HYP.batch_size, shuffle=False)
    recon_loader = DataLoader(recon_dataset, batch_size=1, shuffle=False)

    # model
    model = VAE(ARGS.latent).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HYP.lr)

    if ARGS.pre_train:
        print("Load pre-trained")
        ckpt = os.path.join("model", f"_{ARGS.latent}_{ARGS.epoch}_CVAE_")
        model.load_state_dict(torch.load(ckpt, map_location=device))

    # Original groundtruth mats used later
    org_H = scipy.io.loadmat(PATHS.recon_H_mat)["h"]
    real_H = org_H  # kept same
    geo = scipy.io.loadmat(PATHS.recon_geo_mat)["geo"]

    # Run optimization (your main default behavior)
    z, H = opt(PATHS, xmin, xmax)
    ensure_dirs(PATHS.out_root)
    scipy.io.savemat(os.path.join(PATHS.out_root, "z_allseg.mat"), {"z": z})
    scipy.io.savemat(os.path.join(PATHS.out_root, "H_allseg.mat"), {"H": H})
    scipy.io.savemat(os.path.join(PATHS.out_root, "orgH.mat"), {"real_H": real_H})
