import sys
import json
import time
import argparse

from functools import partial
from sklearn.metrics import mean_squared_error

import torch
import numpy as np
import pandas as pd
import os.path as op

from utils.common import get_device, create_directory
from models import HyperBRDF
from data_processing import MerlDataset, EPFL
from torch.utils.data import DataLoader

def brdf_to_rgb(rvectors, brdf):
    """
    Expects:
      rvectors: [B, N, 6]   -> (hx, hy, hz, dx, dy, dz)
      brdf:     [B, N, 3]   -> per-sample reflectance
    Returns:
      out_rgb:  [B, N, 3]
    """
    hx = rvectors[..., 0]  # shape [B, N]
    hy = rvectors[..., 1]
    hz = rvectors[..., 2]
    dx = rvectors[..., 3]
    dy = rvectors[..., 4]
    dz = rvectors[..., 5]

    theta_h = torch.atan2(torch.sqrt(hx**2 + hy**2), hz)  # [B, N]
    theta_d = torch.atan2(torch.sqrt(dx**2 + dy**2), dz)
    phi_d   = torch.atan2(dy, dx)

    wiz = (torch.cos(theta_d) * torch.cos(theta_h)
           - torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h))  # [B, N]

    # Expand wiz from [B, N] -> [B, N, 1] so it can multiply [B, N, 3]
    wiz_expanded = torch.clamp(wiz, 0, 1).unsqueeze(-1)  # [B, N, 1]
    return brdf * wiz_expanded  # shape [B, N, 3]


def image_mse(model_output, gt):
    # Suppose model_output['model_in'] is [N,6], we do unsqueeze(0) => [1,N,6]
    pred_dirs = model_output['model_in'].unsqueeze(0)
    pred_brdf = model_output['model_out'].unsqueeze(0)
    gt_brdf   = gt['amps'].unsqueeze(0)
    
    # Now pass them to brdf_to_rgb
    rgb_pred = brdf_to_rgb(pred_dirs, pred_brdf)  # => [1,N,3]
    rgb_gt   = brdf_to_rgb(pred_dirs, gt_brdf)
    return {'img_loss': (rgb_pred - rgb_gt).pow(2).mean()}


def latent_loss(model_output):
    """
    Weighted norm of the embedding from set_encoder
    """
    return torch.mean(model_output['latent_vec'] ** 2)

def hypo_weight_loss(model_output):
    """
    Weighted norm of hypernetwork-predicted weights
    """
    weight_sum = 0
    total_weights = 0
    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()
    return weight_sum * (1.0 / total_weights)

def image_hypernetwork_loss(kl, fw, model_output, gt):
    """
    Combine image MSE, latent regularization, and hypernetwork-weight regularization.
    """
    losses = {}
    losses.update(image_mse(model_output, gt))  # 'img_loss'
    losses['latent_loss']      = kl * latent_loss(model_output)
    losses['hypo_weight_loss'] = fw * hypo_weight_loss(model_output)
    return losses

def eval_epoch(model, dataloader, loss_fn, optim, epoch):
    epoch_loss = []
    individual_loss = []
    model.eval()

    with torch.no_grad():
        for step, (model_input, gt) in enumerate(dataloader):
            # Move to device
            model_input = {k: v.to(device) for k,v in model_input.items()}
            gt = {k: v.to(device) for k,v in gt.items()}

            model_output = model(model_input)
            losses = loss_fn(model_output, gt)

            total_loss = 0.
            for _, val in losses.items():
                total_loss += val.item()
            epoch_loss.append(total_loss)
            individual_loss.append([val.item() for val in losses.values()])

    return np.mean(epoch_loss), np.stack(individual_loss).mean(axis=0)

def train_epoch(model, dataloader, loss_fn, optim, epoch):
    epoch_loss = []
    individual_loss = []
    model.train()

    for step, (model_input, gt) in enumerate(dataloader):
        model_input = {k: v.to(device) for k, v in model_input.items()}
        gt = {k: v.to(device) for k, v in gt.items()}

        model_output = model(model_input)
        losses = loss_fn(model_output, gt)

        train_loss = sum(losses.values())
        optim.zero_grad()
        train_loss.backward()

        if clip_grad:
            if isinstance(clip_grad, bool):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optim.step()
        epoch_loss.append(train_loss.item())
        individual_loss.append([v.item() for v in losses.values()])

    return np.mean(epoch_loss), np.stack(individual_loss).mean(axis=0)

def eval_model(model, dataloader, path_=None, name=''):
    """
    Example usage. You can ignore or keep as needed.
    """
    model_inp = []
    model_out = []
    model.eval()

    with torch.no_grad():
        for step, (model_input, gt) in enumerate(dataloader):
            model_input = {k: v.to(device) for k,v in model_input.items()}
            gt = {k: v.to(device) for k,v in gt.items()}
            out = model(model_input)
            model_out.append(out['model_out'].cpu().numpy())
            model_inp.append(gt['amps'].cpu().numpy())

    y_true = np.concatenate(model_inp)[:, :, 0]
    y_pred = np.concatenate(model_out)[:, :, 0]
    return mean_squared_error(y_true, y_pred)


######################################################################################
# MAIN
######################################################################################
parser = argparse.ArgumentParser('')
parser.add_argument('--destdir', dest='destdir', type=str, required=True, help='output directory')
parser.add_argument('--binary', type=str, required=True, help='dataset path')
parser.add_argument('--dataset', choices=['MERL', 'EPFL'], default='MERL')
parser.add_argument('--kl_weight', type=float, default=0., help='latent loss weight')
parser.add_argument('--fw_weight', type=float, default=0., help='hypo loss weight')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--keepon', type=bool, default=False, help='continue training from loaded checkpoint')

args = parser.parse_args()
device = get_device()
print("Running on device:", device)

path_ = op.join(args.destdir, args.dataset)
create_directory(path_)
create_directory(op.join(path_, 'img'))

with open(op.join(path_, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# (CHANGED) Hyperparameters
kl_weight = args.kl_weight
fw_weight = args.fw_weight

loss_fn = partial(image_hypernetwork_loss, kl_weight, fw_weight)
clip_grad = True
lr = args.lr
epochs = args.epochs
binary = args.binary

# (CHANGED) Load dataset that returns { 'coords', 'amps', 'params' }
if args.dataset == 'MERL':
    dataset = MerlDataset(binary)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)
elif args.dataset == 'EPFL':
    dataset = EPFL(binary)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

# (CHANGED) Use the new HyperBRDF that expects [coords, amps, params]
# If your SingleBVPNet uses in_features=6, keep in_features=6 here.
if args.keepon:
    model = torch.load(op.join(path_, 'checkpoint.pt'))
    print("Continuing from existing checkpoint.")
else:
    model = HyperBRDF(in_features=6, out_features=3).to(device)

# Prepare optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr)

# Evaluate initial model
train_losses, all_losses = [], []
init_loss, init_indiv = eval_epoch(model, dataloader, loss_fn, optim, 0)
train_losses.append(init_loss)
all_losses.append(init_indiv)
print(f"Initial Loss = {init_loss}, components = {init_indiv}")

# Train
for epoch in range(epochs):
    e_loss, e_indiv = train_epoch(model, dataloader, loss_fn, optim, epoch)
    train_losses.append(e_loss)
    all_losses.append(e_indiv)
    print(f"Epoch {epoch}/{epochs} - Loss = {e_loss:.6f}")

# Save logs, model
pd.DataFrame(train_losses).to_csv(op.join(path_, 'train_loss.csv'))
pd.DataFrame(all_losses).to_csv(op.join(path_, 'all_losses.csv'))
torch.save(model, op.join(path_, 'checkpoint.pt'))

print("Training complete. Model saved.")
