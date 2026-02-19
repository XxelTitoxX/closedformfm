#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tsteps", type=int, default=21, help="Number of time steps in [0,0.99]")
    parser.add_argument("--batch_size_x1prime", type=int, default=256, help="Batch size for x1' bank")
    parser.add_argument("--n_subsample", type=int, default=5000, help="Number of CIFAR-10 train samples to use")
    parser.add_argument("--sigmamin", type=float, default=0.0, help="Sigma_min for u*")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out_dir", default="experiments/outputs", help="Output directory")
    parser.add_argument("--x1_index", type=int, default=0, help="Index of x1 in the subsample")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

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

    if not (0 <= args.x1_index < len(dataset)):
        raise ValueError(f"x1_index must be in [0, {len(dataset)-1}]")

    # Choose x1 from the subsample and x0 from noise
    x1, _ = dataset[args.x1_index]
    x1 = x1.unsqueeze(0)
    x0 = torch.randn_like(x1)

    # Build x1' bank loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size_x1prime,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    t_values = torch.linspace(0.0, 0.99, args.n_tsteps)
    q25 = []
    q50 = []
    q75 = []
    w_x1 = []

    for t in t_values:
        t_batch = t.view(1, 1)
        t_img = t.view(1, 1, 1, 1)
        xt = (1 - t_img) * x0 + t_img * x1

        # Collect logits for all x1'
        logits = []
        for batch in loader:
            x1prime = batch[0]
            x1prime_flat = torch.flatten(x1prime, start_dim=1)
            xt_flat = torch.flatten(xt, start_dim=1)
            mut = t_batch * x1prime_flat
            sigmat = (1 - (1 - args.sigmamin) * t_batch)
            arg = -((mut - xt_flat) ** 2).sum(dim=-1) / (2 * sigmat.squeeze() ** 2)
            logits.append(arg)
        logits = torch.cat(logits, dim=0)

        weights = torch.softmax(logits, dim=0).cpu().numpy()
        q25.append(np.quantile(weights, 0.25))
        q50.append(np.quantile(weights, 0.50))
        q75.append(np.quantile(weights, 0.75))
        w_x1.append(weights[args.x1_index])

    q25 = np.array(q25)
    q50 = np.array(q50)
    q75 = np.array(q75)
    w_x1 = np.array(w_x1)

    out_prefix = os.path.join(
        args.out_dir,
        f"weights_t{args.n_tsteps}_n{args.n_subsample}_bs{args.batch_size_x1prime}_x1{args.x1_index}",
    )
    np.save(
        out_prefix + ".npy",
        {"t": t_values.numpy(), "q25": q25, "q50": q50, "q75": q75, "w_x1": w_x1},
    )

    plt.figure(figsize=(6, 4))
    plt.plot(t_values, q50, label="median")
    plt.fill_between(t_values, q25, q75, alpha=0.2, label="q25–q75")
    plt.plot(t_values, w_x1, marker="o", linestyle="--", label="w(x1)")
    plt.xlabel("t")
    plt.ylabel("weight")
    plt.title("Weights for $u_*(x_t, t)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + ".png", dpi=150)
    print(f"Saved {out_prefix}.png")


if __name__ == "__main__":
    main()
