#!/usr/bin/env python3
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision import datasets, transforms

from torchcfm.models.unet.unet import UNetModelWrapper
from utils.mean_cfm import get_full_velocity_field, get_full_velocity_field_from_loader
import contextlib
import io


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path, num_channels, device, use_ema=True):
    model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_model" if use_ema and "ema_model" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model

def plot_from_csv(csv_path, fig_path):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t_vals = data[:, 0]
    mse_vals = data[:, 1]

    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, mse_vals, marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\mathbb{E}[\|u_\theta(x,t) - u_*(x,t)\|^2]$")
    plt.ylim(bottom=0, top=1)
    plt.title("Velocity Field Approximation Error")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out_dir", default="experiments/outputs", help="Output directory")
    parser.add_argument("--n_pairs", type=int, default=1024, help="Number of (x0, x1) pairs")
    parser.add_argument("--n_tsteps", type=int, default=21, help="Number of time steps in [0,1]")
    parser.add_argument("--batch_size_xt", type=int, default=128, help="Batch size for x0/x1 (x_t) pairs")
    parser.add_argument("--batch_size_x1prime", type=int, default=128, help="Batch size for x1' samples")
    parser.add_argument("--num_channels", type=int, default=128, help="UNet channels")
    parser.add_argument("--sigmamin", type=float, default=0.0, help="Sigma_min for u*")
    parser.add_argument("--use_ema", action="store_true", help="Use ema_model from checkpoint")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_subsample", type=int, default=None, help="Number of CIFAR-10 train samples to use")
    parser.add_argument("--ustar_n_samples", type=int, default=None, help="Number of x1' samples used for u*")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    # device = "cpu"

    os.makedirs(args.out_dir, exist_ok=True)

    fig_path = os.path.join(args.out_dir, "velocity_error.png")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root=os.path.expanduser("~/datasets"),
        train=True,
        download=True,
        transform=transform,
    )
    if args.n_subsample is not None:
        dataset.data = dataset.data[: args.n_subsample]
        if hasattr(dataset, "targets"):
            dataset.targets = dataset.targets[: args.n_subsample]

    model = load_model(args.checkpoint, args.num_channels, device, use_ema=args.use_ema)

    t_values = torch.linspace(0.0, 0.99, args.n_tsteps, device=device)
    mse_sums = torch.zeros(args.n_tsteps, device=device)
    counts = torch.zeros(args.n_tsteps, device=device)

    num_batches = int(np.ceil(args.n_pairs / args.batch_size_xt))
    xt_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size_xt, shuffle=True, drop_last=True
    )
    x1prime_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size_x1prime, shuffle=True, drop_last=True
    )
    xt_iter = iter(xt_loader)

    start = time.time()
    for _ in tqdm(range(num_batches), desc="Batches", dynamic_ncols=True):
        try:
            x1, _ = next(xt_iter)
        except StopIteration:
            xt_iter = iter(xt_loader)
            x1, _ = next(xt_iter)

        x1 = x1.to(device)
        x0 = torch.randn_like(x1)

        for i, t in enumerate(t_values):
            t_batch = t.expand(x0.shape[0], 1)
            t_img = t_batch.view(-1, 1, 1, 1)
            xt = (1 - t_img) * x0 + t_img * x1

            with torch.no_grad():
                with contextlib.redirect_stdout(io.StringIO()):
                    u_theta = model(t_batch, xt)
                    u_star = get_full_velocity_field_from_loader(
                        t_batch, xt, x1prime_loader,
                        sigmamin=args.sigmamin,
                        n_samples=args.ustar_n_samples,
                        flatten=True,
                    )

            diff = (u_theta - u_star) ** 2
            mse = diff.view(diff.shape[0], -1).mean(dim=1).sum()
            mse_sums[i] += mse
            counts[i] += diff.shape[0]

    mse_vals = (mse_sums / counts).detach().cpu().numpy()
    t_vals = t_values.detach().cpu().numpy()

    fig_title = f"velocity_error_{args.checkpoint.replace("/", "_")}_train{args.n_subsample}_t{args.n_tsteps}_samp{args.n_pairs}_{args.batch_size_x1prime}"
    npy_path = os.path.join(args.out_dir, fig_title+".npy")
    csv_path = os.path.join(args.out_dir, fig_title+".csv")
    fig_path = os.path.join(args.out_dir, fig_title+".png")

    np.save(npy_path, {"t": t_vals, "mse": mse_vals})
    np.savetxt(csv_path, np.stack([t_vals, mse_vals], axis=1), delimiter=",", header="t,mse", comments="")

    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, mse_vals, marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\mathbb{E}[\|u_\theta(x,t) - u_*(x,t)\|^2]$")
    plt.ylim(bottom=0, top=3)
    plt.title("Velocity Field Approximation Error")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)

    elapsed = time.time() - start
    print(f"Saved {fig_path}")
    print(f"Saved {csv_path}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    #main()
    plot_from_csv("experiments/outputs/velocity_error_checkpoints_20260219_100101_n5000_CFM_noflip_bias_step_50000.pt_train5000_t21_samp1024_128.csv", "experiments/outputs/velocity_error_n5000_CFM_noflip_bias.png")
