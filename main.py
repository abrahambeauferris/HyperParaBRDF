#!/usr/bin/env python3

import sys
import json
import time
import argparse
import os.path as op
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Save figures without an X server

from functools import partial
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from utils.common import get_device, create_directory
from models import HyperBRDF
from data_processing import MerlDataset, EPFL

############################################################################
# Original losses: image_hypernetwork_loss and dependencies
############################################################################

def brdf_to_rgb(rvectors, brdf):
    hx = rvectors[..., 0]
    hy = rvectors[..., 1]
    hz = rvectors[..., 2]
    dx = rvectors[..., 3]
    dy = rvectors[..., 4]
    dz = rvectors[..., 5]

    theta_h = torch.atan2(torch.sqrt(hx**2 + hy**2), hz)
    theta_d = torch.atan2(torch.sqrt(dx**2 + dy**2), dz)
    phi_d   = torch.atan2(dy, dx)

    wiz = (torch.cos(theta_d)*torch.cos(theta_h)
           - torch.sin(theta_d)*torch.cos(phi_d)*torch.sin(theta_h))
    wiz_expanded = torch.clamp(wiz, 0, 1).unsqueeze(-1)
    return brdf * wiz_expanded

def image_mse(model_output, gt):
    pred_dirs = model_output['model_in'].unsqueeze(0)
    pred_brdf = model_output['model_out'].unsqueeze(0)
    gt_brdf   = gt['amps'].unsqueeze(0)

    rgb_pred = brdf_to_rgb(pred_dirs, pred_brdf)
    rgb_gt   = brdf_to_rgb(pred_dirs, gt_brdf)
    return {'img_loss': (rgb_pred - rgb_gt).pow(2).mean()}

def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)

def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0
    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()
    return weight_sum * (1.0 / total_weights)

def image_hypernetwork_loss(kl, fw, model_output, gt):
    losses = {}
    losses.update(image_mse(model_output, gt))  # 'img_loss'
    losses['latent_loss']      = kl * latent_loss(model_output)
    losses['hypo_weight_loss'] = fw * hypo_weight_loss(model_output)
    return losses

############################################################################
# Training / Evaluation Routines
############################################################################

def train_epoch(model, dataloader, loss_fn, optim, device, clip_grad=True):
    model.train()
    epoch_loss = []
    indiv_loss = []

    for model_input, gt in dataloader:
        model_input = {k: v.to(device) for k, v in model_input.items()}
        gt          = {k: v.to(device) for k, v in gt.items()}

        model_output = model(model_input)
        losses_dict  = loss_fn(model_output, gt)

        total_loss = sum(losses_dict.values())

        optim.zero_grad()
        total_loss.backward()

        if clip_grad:
            if isinstance(clip_grad, bool):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

        optim.step()
        epoch_loss.append(total_loss.item())
        indiv_loss.append([l.item() for l in losses_dict.values()])

    return np.mean(epoch_loss), np.stack(indiv_loss).mean(axis=0)

def eval_epoch(model, dataloader, loss_fn, device):
    model.eval()
    epoch_loss = []
    indiv_loss = []

    with torch.no_grad():
        for model_input, gt in dataloader:
            model_input = {k: v.to(device) for k, v in model_input.items()}
            gt          = {k: v.to(device) for k, v in gt.items()}

            model_output = model(model_input)
            losses_dict  = loss_fn(model_output, gt)

            total_loss = sum(l.item() for l in losses_dict.values())
            epoch_loss.append(total_loss)
            indiv_loss.append([l.item() for l in losses_dict.values()])

    return np.mean(epoch_loss), np.stack(indiv_loss).mean(axis=0)

def compute_rmse_mae(model, dataloader, device):
    """
    Evaluate final predictions vs. ground truth (RMSE, MAE).
    We'll compare 'model_out' vs. 'amps' across entire dataset subset.
    """
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for model_input, gt in dataloader:
            model_input = {k: v.to(device) for k,v in model_input.items()}
            gt          = {k: v.to(device) for k,v in gt.items()}
            out = model(model_input)
            # out['model_out'] => shape [N,3], gt['amps'] => shape [N,3]
            preds.append(out['model_out'].cpu().numpy())
            trues.append(gt['amps'].cpu().numpy())

    Y_pred = np.concatenate(preds, axis=0).reshape(-1)
    Y_true = np.concatenate(trues, axis=0).reshape(-1)

    rmse_val = mean_squared_error(Y_true, Y_pred, squared=False)
    mae_val  = mean_absolute_error(Y_true, Y_pred)
    return rmse_val, mae_val

############################################################################
# Main
############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destdir',    type=str, required=True, help='output directory')
    parser.add_argument('--binary',     type=str, required=True, help='dataset path')
    parser.add_argument('--dataset',    choices=['MERL', 'EPFL'], default='MERL')
    parser.add_argument('--kl_weight',  type=float, default=0.,  help='latent loss weight')
    parser.add_argument('--fw_weight',  type=float, default=0.,  help='hypo loss weight')
    parser.add_argument('--epochs',     type=int,   default=80,  help='number of epochs')
    parser.add_argument('--lr',         type=float, default=5e-5,help='learning rate')
    parser.add_argument('--keepon',     type=bool,  default=False, help='continue from checkpoint?')

    # e.g. 0.0 => means "train on the full dataset"
    parser.add_argument('--test_sizes', type=float, nargs='+', default=[0.0,0.1,0.2,0.3,0.4,0.5],
                        help='List of test splits for thorough evaluation')

    args = parser.parse_args()
    device = get_device()
    print("Running on device:", device)

    base_path = op.join(args.destdir, args.dataset)
    create_directory(base_path)

    # Save arguments for reference
    with open(op.join(base_path, 'args.txt'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Prepare data
    if args.dataset == 'MERL':
        full_dataset = MerlDataset(args.binary)
    else:
        full_dataset = EPFL(args.binary)

    all_idx   = np.arange(len(full_dataset))
    kl_weight = args.kl_weight
    fw_weight = args.fw_weight
    loss_fn   = partial(image_hypernetwork_loss, kl_weight, fw_weight)

    # We'll store summary of each run in memory, then write out
    summary_records = []

    for tsize in args.test_sizes:
        # Create a subfolder
        run_path = op.join(base_path, f"testsize_{tsize}")
        create_directory(run_path)

        # If tsize == 0 => train on entire data, no test
        if tsize > 0.0:
            train_idx, test_idx = train_test_split(all_idx, test_size=tsize, random_state=42)
            train_subset = Subset(full_dataset, train_idx)
            test_subset  = Subset(full_dataset, test_idx)
            print(f"\n=== Split test_size={tsize:.2f} => Train={len(train_idx)} | Test={len(test_idx)} ===")

        else:
            # full dataset => no test
            train_subset = full_dataset
            test_subset  = None
            print(f"\n=== No test split => training on entire dataset. (tsize={tsize}) ===")

        train_loader = DataLoader(train_subset, shuffle=True, batch_size=1)
        if test_subset is not None:
            test_loader  = DataLoader(test_subset, shuffle=False, batch_size=1)

        # If keepon, try to load checkpoint
        ckpt_path = op.join(run_path, 'checkpoint.pt')
        if args.keepon and op.exists(ckpt_path):
            model = torch.load(ckpt_path, map_location=device)
            print("Continuing from existing checkpoint for test_size=", tsize)
        else:
            model = HyperBRDF(in_features=6, out_features=3).to(device)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Evaluate initial train loss
        init_loss, init_indiv = eval_epoch(model, train_loader, loss_fn, device)
        print(f"[Split={tsize}] Initial train loss = {init_loss:.6f}")

        train_losses = [init_loss]

        # Train
        for ep in range(args.epochs):
            e_loss, e_ind = train_epoch(model, train_loader, loss_fn, optim, device, clip_grad=True)
            train_losses.append(e_loss)
            if (ep+1) % 10 == 0:
                print(f"  Epoch {ep+1}/{args.epochs} - train_loss={e_loss:.6f}")

        # Save logs
        pd.DataFrame(train_losses, columns=['train_loss']).to_csv(op.join(run_path, 'train_loss.csv'), index=False)
        torch.save(model, ckpt_path)

        # Evaluate on test set if we have one
        if test_subset is not None:
            rmse_val, mae_val = compute_rmse_mae(model, test_loader, device)
            print(f"[Split={tsize}] Final Test RMSE={rmse_val:.5f}, MAE={mae_val:.5f}")
            summary_records.append((tsize, len(train_subset), len(test_subset), rmse_val, mae_val))
        else:
            # no test => record something
            summary_records.append((tsize, len(train_subset), 0, np.nan, np.nan))

        # Plot
        plt.figure()
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.title(f"Train Loss (test_size={tsize})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        out_svg = op.join(run_path, "train_loss_plot.svg")
        plt.savefig(out_svg, format='svg', dpi=300)
        plt.close()

    # Summaries
    columns = ['test_size','train_count','test_count','rmse','mae']
    df_sum = pd.DataFrame(summary_records, columns=columns)
    df_sum.to_csv(op.join(base_path, "test_splits_summary.csv"), index=False)

    # Also produce a text-based summary
    txt_path = op.join(base_path, "comprehensive_results.txt")
    with open(txt_path, 'w') as f:
        f.write("====== Comprehensive Results ======\n")
        f.write("test_size  |  train_count  |  test_count  |   RMSE     |   MAE\n")
        for row in summary_records:
            tsize, trn, tst, rm, ma = row
            f.write(f"{tsize:<10.2f} | {trn:<12d} | {tst:<10d} | {rm:<10.5f} | {ma:<10.5f}\n")
        f.write("\n(Done)\n")

    print("\n====== Final Summaries ======")
    print(df_sum)
    print(f"\nWrote results to {base_path}\n")

if __name__ == "__main__":
    main()
