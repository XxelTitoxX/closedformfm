# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.
# and https://github.com/atong01/conditional-flow-matching/blob/main/examples/images/cifar10/train_cifar10.py

import time
import copy
import math

from utils.core import hydra_main, mlflow_start
from utils.sampling_strategy import uniform_sampling, biased_sampling

total_num_gpus = 1


num_workers = 2


# delay most imports to have faster access to hydra config from the command line

@hydra_main("train_cifar10")
def train(cfg):
    def warmup_lr(step):
        return min(step, warmup) / warmup

    import numpy as np
    import torch
    import contextlib
    import io
    import os
    from torchvision import datasets, transforms
    from tqdm.auto import trange
    from utils.cifar10 import ema, infiniteloop, model_mul
    # generate_samples
    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher,
        TargetConditionalFlowMatcher,
        VariancePreservingConditionalFlowMatcher,
        pad_t_like_x
    )
    from torchcfm.models.unet.unet import UNetModelWrapper
    from utils.metrics import getall
    from utils.mean_cfm import get_full_velocity_field

    from torchvision.datasets.cifar import CIFAR10
    from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
    from utils.compute_fid import compute_fid

    import mlflow
    from mlflow.models import infer_signature
    print("Training about to start")

    mlflow_start(cfg, "cifar10")
    
    print("Training started")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    num_channels = getall(cfg.net, 'num_channels')[0]
    batch_size, lr, warmup, ema_decay, ema_start, grad_clip = getall(
        cfg.optimizer, 'batch_size, lr, warmup, ema_decay, ema_start, grad_clip')
    (total_steps, dump_models, dump_every, dump_points, nodump_before,
     compute_fid5k_every, compute_fid50k_every, save_model_every,
     batch_size_fid, silent_tqdm, compute_pr, pr_k, pr_subset_size,
     pr_seed) = getall(
        cfg.trainer,
        'total_steps, dump_models, dump_every, dump_points, nodump_before, '
        'compute_fid5k_every, compute_fid50k_every, save_model_every, '
        'batch_size_fid, silent_tqdm, compute_pr, pr_k, pr_subset_size, '
        'pr_seed')
    expected_ucond, n_samples_mean, batch_size_mean, sigmamin, model_name, rescaled, tmin, tmax = getall(
        cfg.loss,
        'expected_ucond, n_samples_mean, batch_size_mean, sigmamin, model_name, rescaled, tmin, tmax')
    integration_method, integration_steps = getall(
        cfg.sampler, "integration_method, integration_steps")
    root, n_subsample, random_horizontal_flip = getall(
        cfg.data, "root, n_subsample, random_horizontal_flip")
    print(f"Number CIFAR-10 subsamples: {n_subsample}")

    torch.manual_seed(0)  # TODO better seeding

    if random_horizontal_flip:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    # DATASETS/DATALOADER
    # import ipdb; ipdb.set_trace()
    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    if n_subsample < 50_000:
        dataset.data = dataset.data[:n_subsample]
    batch_size = min(batch_size, n_subsample)

    ft_extractor = InceptionFeatureExtractor(save_path="features")
    train_feat = ft_extractor.get_features(
        CIFAR10(train=True, root=root, download=True), name="cifar10_train")

    sampler = None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    datalooper = infiniteloop(dataloader)

    if expected_ucond:
        dataloader_prime = torch.utils.data.DataLoader(
            dataset,
            batch_size=n_samples_mean,
            sampler=sampler,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        datalooper_prime = infiniteloop(dataloader_prime)

    # Calculate number of epochs
    steps_per_epoch = math.ceil(len(dataset) / batch_size)
    num_epochs = math.ceil((total_steps + 1) / steps_per_epoch)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=num_channels,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(
        device
    )

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    # sigma = 0.0
    if model_name == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigmamin)
    elif model_name == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigmamin)
    elif model_name == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigmamin)
    elif model_name == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigmamin)
    else:
        raise NotImplementedError(
            f"Unknown {model_name}, must be in ['otcfm', 'icfm', 'fm', 'si']")

    global_step = 0  # to keep track of the global step in training loop
    best_fid = float("inf")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    checkpoints_root = os.path.join(project_root, "checkpoints")
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoints_root, f"{run_stamp}_n{n_subsample}")
    inter_dir = os.path.join(run_dir, "intermediate")
    os.makedirs(inter_dir, exist_ok=True)

    for idx_epoch in range(num_epochs):
        epoch_start = time.time()
        if sampler is not None:
            sampler.set_epoch(idx_epoch)

        if silent_tqdm:
            step_iter = range(steps_per_epoch)
            step_pbar = None
        else:
            step_pbar = trange(
                steps_per_epoch,
                dynamic_ncols=True,
                desc=f"Epoch {idx_epoch + 1}/{num_epochs}",
            )
            step_iter = step_pbar
        for step in step_iter:
                    start = time.time()

                    # Silence noisy prints from dependencies during batch compute.
                    with contextlib.redirect_stdout(io.StringIO()):
                        optim.zero_grad()
                        x1 = next(datalooper).to(device)
                        x0 = torch.randn_like(x1)

                        if cfg.t_sampling.strategy == "uniform":
                            t = uniform_sampling(x0, tmin, tmax)
                        elif cfg.t_sampling.strategy == "biased":
                            t = biased_sampling(x0, tmin, tmax, tc=cfg.t_sampling.tc, bump_weight=cfg.t_sampling.bump_weight, bump_width=cfg.t_sampling.bump_width)
                        t_ = pad_t_like_x(t, x0)
                        xt = (1 - t_) * x0 + t_ * x1
                        if expected_ucond and model_name == "icfm":
                            # TODO larger batchsize for x1_prime
                            x1prime = next(datalooper_prime).to(device)
                            x1prime = torch.cat([x1, x1prime])
                            ut = get_full_velocity_field(
                                t, xt, x1prime, sigmamin=sigmamin,
                                batch_size_mean=batch_size_mean, flatten=True)
                        elif (not expected_ucond) and model_name == "icfm":
                            ut = x1 - x0
                        else:
                            t, xt, ut = FM.sample_location_and_conditional_flow(
                                x0, x1)

                        vt = net_model(t, xt)
                        if rescaled:
                            t_ = pad_t_like_x(t, x0)
                            loss = torch.mean(((vt - ut) * (1 - t_)) ** 2)
                        loss = torch.mean((vt - ut) ** 2)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            net_model.parameters(), grad_clip)  # new
                        optim.step()
                        sched.step()

                    # We have done one more step, increment here so that e.g. after 50000 backward, we have global_step%50000 == 0, etc
                    global_step += 1

                    # Update EMA model, ema_start>0 to use corrected one (and skip the first steps)
                    if ema_start == 0: # default ema, actually somewhat wrong, too much weight on the first model
                        ema(net_model, ema_model, ema_decay)
                    else:
                        if global_step == ema_start:
                            unbalanced_ema_model = copy.deepcopy(net_model)
                            model_mul(1-ema_decay, net_model, unbalanced_ema_model) # unb = (1-ema_decay) * M0
                        elif global_step > ema_start:
                            ema(net_model, unbalanced_ema_model, ema_decay)
                            _factor = 1 / (1 - ema_decay**(1 + global_step - ema_start))
                            model_mul(_factor, unbalanced_ema_model, ema_model)
                    end = time.time()

                    with torch.no_grad():
                        if global_step % 100 == 0:
                            mlflow.log_metric(
                                "Training Loss", loss, step=global_step)
                            mlflow.log_metric(
                                "Gradient Norm", grad_norm, step=global_step)
                        if step_pbar is not None:
                            step_pbar.set_postfix(
                                loss=f"{loss.item():0.4f}",
                                time=f"{(end - start):0.2f}s",
                                step=global_step,
                            )
                        if (global_step % compute_fid5k_every == 10):# and global_step > 0:
                            # Compute FID 5k online
                            num_gen = 5_000
                            fid_out = compute_fid(
                                ema_model, num_gen, train_feat,
                                device, ft_extractor, batch_size_fid,
                                integration_method=integration_method,
                                integration_steps=integration_steps,
                                desc=f"FID {num_gen // 1000}k (gen)",
                                compute_pr=compute_pr,
                                pr_k=pr_k,
                                pr_subset_size=pr_subset_size,
                                pr_seed=pr_seed)
                            if compute_pr:
                                fid, precision, recall, gen_metric = fid_out
                                mlflow.log_metric(
                                    f"Precision - {(num_gen // 1000)}k {integration_method} {integration_steps} steps",
                                    precision, step=global_step)
                                mlflow.log_metric(
                                    f"Recall - {(num_gen // 1000)}k {integration_method} {integration_steps} steps",
                                    recall, step=global_step)
                                mlflow.log_metric(
                                    f"GenMetric - {(num_gen // 1000)}k {integration_method} {integration_steps} steps",
                                    gen_metric, step=global_step)
                            else:
                                fid = fid_out
                            metric_title = f"FID - {(num_gen // 1000)}k {integration_method} {integration_steps} steps"
                            mlflow.log_metric(
                                metric_title, fid, step=global_step)
                            if step_pbar is not None:
                                step_pbar.write(f"{metric_title}: {fid:.4f}")
                            else:
                                print(f"{metric_title}: {fid:.4f}")
                            if fid < best_fid:
                                best_fid = fid
                                torch.save(
                                    {
                                        "ema_model": ema_model.state_dict(),
                                        "model": net_model.state_dict(),
                                        "optimizer": optim.state_dict(),
                                        "scheduler": sched.state_dict(),
                                        "global_step": global_step,
                                        "fid": float(fid),
                                    },
                                    os.path.join(run_dir, "best.pt"),
                                )

                        if (np.log2(global_step).is_integer() or global_step % 50_000 == 0) and global_step >= nodump_before:
                            # import ipdb; ipdb.set_trace()
                            signature = infer_signature(
                                x0.cpu().numpy(),
                                ema_model(t, x0).detach().cpu().numpy())
                            mlflow.pytorch.log_model(
                                ema_model, name=f'model_ema_{global_step}',
                                signature=signature)
                            mlflow.pytorch.log_model(
                                net_model, name=f'model_{global_step}',
                                signature=signature)
        if step_pbar is not None:
            step_pbar.close()
        torch.save(
            {
                "ema_model": ema_model.state_dict(),
                "model": net_model.state_dict(),
                "optimizer": optim.state_dict(),
                "scheduler": sched.state_dict(),
                "global_step": global_step,
            },
            os.path.join(run_dir, "latest.pt"),
        )
        if (global_step % save_model_every) == 0:
            torch.save(
                {
                    "ema_model": ema_model.state_dict(),
                    "model": net_model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "scheduler": sched.state_dict(),
                    "global_step": global_step,
                    "epoch": idx_epoch + 1,
                },
                os.path.join(inter_dir, f"step_{global_step}.pt"),
            )
        epoch_elapsed = time.time() - epoch_start
        print(f"Epoch {idx_epoch + 1}/{num_epochs}, elapsed time={epoch_elapsed:.2f}s, global step={global_step}, best FID={best_fid:.4f}, last loss={loss.item():.4f}")


if __name__ == "__main__":
    train()
