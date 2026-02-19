#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from torchcfm.models.unet.unet import UNetModelWrapper
from utils.metrics import generate_samples


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


def to_uint8(imgs):
    imgs = imgs.clamp(-1, 1)
    imgs = (imgs * 0.5 + 0.5) * 255.0
    return imgs.to(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out_dir", default="experiments/outputs", help="Output directory")
    parser.add_argument("--num_channels", type=int, default=128, help="UNet channels")
    parser.add_argument("--use_ema", action="store_true", help="Use ema_model from checkpoint")
    parser.add_argument("--n_subsample", type=int, default=None, help="Number of CIFAR-10 train samples to use")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--integration_method", default="euler", help="ODE solver")
    parser.add_argument("--integration_steps", type=int, default=10, help="ODE steps")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
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

    model = load_model(args.checkpoint, args.num_channels, device, use_ema=args.use_ema)

    with torch.no_grad():
        gen = generate_samples(
            model,
            device,
            integration_method=args.integration_method,
            integration_steps=args.integration_steps,
            n_samples=3,
            batch_size=3,
            res=32,
            desc="Generate",
        )

    gen_uint8 = to_uint8(gen).cpu()
    gen_flat = gen_uint8.float().view(3, -1)

    # Prepare dataset tensor for nearest neighbor search
    ds_imgs = torch.stack([dataset[i][0] for i in range(len(dataset))])
    ds_uint8 = to_uint8(ds_imgs).cpu()
    ds_flat = ds_uint8.float().view(ds_uint8.shape[0], -1)

    # Compute nearest neighbors (L2)
    dists = torch.cdist(gen_flat, ds_flat)
    nn_idx = torch.argmin(dists, dim=1)
    nn_imgs = ds_uint8[nn_idx]

    # Plot: 2 rows (generated, nearest neighbor), 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for i in range(3):
        axes[0, i].imshow(gen_uint8[i].permute(1, 2, 0))
        axes[0, i].set_title(f"Generated {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(nn_imgs[i].permute(1, 2, 0))
        axes[1, i].set_title(f"Nearest Neighbor {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "samples_and_nn.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
