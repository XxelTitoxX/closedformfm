import torch
import numpy as np
import os
import hydra
import mlflow
import mlflow.pytorch as mlpt
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.celeba import CelebA
from fld.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from fld.metrics.FID import FID
from mlflow import MlflowClient
from torchvision import datasets, transforms

from utils.metrics import generate_samples

EXTRACTORS = {"inception": InceptionFeatureExtractor, "dino": DINOv2FeatureExtractor}


def _to_torch(feat):
    if isinstance(feat, torch.Tensor):
        return feat.detach().cpu()
    return torch.from_numpy(feat)


def _knn_radius(x, k, batch_size=1024):
    n = x.shape[0]
    if k >= n:
        raise ValueError(f"k={k} must be smaller than number of samples {n}")
    radii = torch.empty(n, dtype=x.dtype)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        d = torch.cdist(x[start:end], x)
        # Exclude self distances
        rows = torch.arange(start, end)
        d[torch.arange(end - start), rows] = float("inf")
        knn = torch.topk(d, k, largest=False).values[:, -1]
        radii[start:end] = knn
    return radii


def _precision_recall(train_feat, gen_feat, k=3, subset_size=5000, seed=0):
    rng = np.random.default_rng(seed)
    train_feat = _to_torch(train_feat)
    gen_feat = _to_torch(gen_feat)

    n_train = train_feat.shape[0]
    n_gen = gen_feat.shape[0]
    n_sub = min(subset_size, n_train)
    idx = rng.choice(n_train, size=n_sub, replace=False)
    train_sub = train_feat[idx]

    k = min(k, n_sub - 1, n_gen - 1)
    if k <= 0:
        raise ValueError("Not enough samples to compute precision/recall.")

    radius_train = _knn_radius(train_sub, k)
    radius_gen = _knn_radius(gen_feat, k)

    # Precision: fraction of gen within train manifold
    prec_hits = 0
    for start in range(0, n_gen, 512):
        end = min(start + 512, n_gen)
        d = torch.cdist(gen_feat[start:end], train_sub)
        hit = (d <= radius_train[None, :]).any(dim=1)
        prec_hits += hit.sum().item()
    precision = prec_hits / n_gen

    # Recall: fraction of train within gen manifold
    rec_hits = 0
    for start in range(0, n_sub, 512):
        end = min(start + 512, n_sub)
        d = torch.cdist(train_sub[start:end], gen_feat)
        hit = (d <= radius_gen[None, :]).any(dim=1)
        rec_hits += hit.sum().item()
    recall = rec_hits / n_sub

    return precision, recall, train_sub, gen_feat


def _generalization_metric(gen_feat, train_sub):
    n_gen = gen_feat.shape[0]
    ratios = []
    for start in range(0, n_gen, 512):
        end = min(start + 512, n_gen)
        d = torch.cdist(gen_feat[start:end], train_sub)
        nn2 = torch.topk(d, 2, largest=False).values
        ratio = (nn2[:, 0] / (nn2[:, 1] + 1e-12)).cpu().numpy()
        ratios.append(ratio)
    return float(np.mean(np.concatenate(ratios)))


def compute_fid(model, num_images_fid, train_feat, device, ft_extractor, batch_size=512, integration_method="dopri5", integration_steps=100, res=32, desc=None, compute_pr=False, pr_k=3, pr_subset_size=5000, pr_seed=0):
    gen_images = generate_samples(model, device, integration_method=integration_method, tol=1e-4,
                                  n_samples=num_images_fid, res=res, batch_size=batch_size, integration_steps=integration_steps, desc=desc)
    gen_images = (gen_images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    gen_feat = ft_extractor.get_tensor_features(
        gen_images)
    fid_val = FID().compute_metric(
        train_feat, None, gen_feat)
    if not compute_pr:
        return fid_val
    precision, recall, train_sub, gen_feat_t = _precision_recall(
        train_feat, gen_feat, k=pr_k, subset_size=pr_subset_size, seed=pr_seed)
    gen_metric = _generalization_metric(gen_feat_t, train_sub)
    return fid_val, precision, recall, gen_metric


def compute_fid_from_dir(gen_path, num_images_fid, train_feat, device, ft_extractor):
    gen_feat = ft_extractor.get_dir_features(gen_path)
    fid_val = FID().compute_metric(
        train_feat, None, gen_feat)
    return fid_val


@hydra.main(version_base=None, config_path="conf", config_name="config_metric")
def log_fid(cfg):
    # run_id = "8efb498ce85e4afab1d3292a4a3674c7"
    run_id = cfg.run_id
    ema = cfg.generation.ema

    save_path = (f"samples/{run_id}/solver_{cfg.generation.solver}/"
                 f"inte_steps_{cfg.generation.steps}/ema_{cfg.generation.ema}/"
                 f"{cfg.fid.num_images_fid//1000}k")

    mlflow.start_run(run_id)
    client = mlflow.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    ft_extractor = EXTRACTORS[cfg.fid.ft_extractor](save_path="features")

    if cfg.dataset.lower() == "cifar10":
        train_feat = ft_extractor.get_features(
            CIFAR10(train=True, root="data", download=True), name="cifar10_train")
        test_feat = ft_extractor.get_features(
            CIFAR10(train=False, root="data", download=True), name="cifar10_test")
        res = 32
    elif cfg.dataset.lower() == "celeba":
        train_feat = ft_extractor.get_features(
            CelebA(split='train', root="../data", download=True,     transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="celeba64_train")
        test_feat = ft_extractor.get_features(
            CelebA(split='test', root="../data", download=True, transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="celeba64_test")
        res = 64
    elif cfg.dataset.lower() == "tiny_imagenet":
        from tinyimagenet import TinyImageNet # import on demand to get optional dependency
        train_feat = ft_extractor.get_features(
            TinyImageNet(split='train', root="./data/tinyimagenet",    transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="tiny_train")
        test_feat = ft_extractor.get_features(
            TinyImageNet(split='test', root="./data/tinyimagenet", transform=transforms.Compose(
                [transforms.CenterCrop(178),
                 transforms.Resize([64, 64]),])), name="tiny_test")
        res = 64

    else:
        raise ValueError()
    # test_feat = ft_extractor.get_features(
    # CIFAR10(train=False, root="data", download=True))

    num_images_fid = cfg.fid.num_images_fid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_steps = cfg.generation.global_steps
    print(f"{global_steps=}")
    if len(global_steps) == 0:
        global_steps = []
        # we expect model and model_ema to be saved at same global steps:
        for artifact in artifacts:
            if not artifact.path.startswith("model_"):
                continue
            else:
                if not artifact.path.startswith("model_ema"):
                    global_steps.append(int(artifact.path[len("model_"):]))
        global_steps = np.sort(global_steps)

    for global_step in global_steps:
        print("k", global_step)
        path = f"runs:/{run_id}/model_{'ema_'*ema}{global_step}"

        gen_path = save_path + f"/step_{global_step}/"
        # fid_val = compute_fid(model, num_images_fid, train_feat, device, ft_extractor)
        # be carfefull to the number of samples in the dir
        if os.path.isdir(gen_path) and len(os.listdir(gen_path)) == num_images_fid:
            pass
        else:
            model = mlpt.load_model(path)
            model.eval()
            model.to(device)
            print(f"Generating {num_images_fid} samples for FID")
            generate_samples(model, device, path=gen_path, integration_method=cfg.generation.solver, tol=1e-4,
                             n_samples=num_images_fid, batch_size=cfg.generation.batch_size_gen, integration_steps=cfg.generation.steps, res=res)

        for data_name, feat in zip(["train", "test"], [train_feat, test_feat]):
            fid = compute_fid_from_dir(
                gen_path, num_images_fid, feat, device, ft_extractor)
            metric_name = (
                f"FID {data_name} {cfg.fid.ft_extractor} {cfg.fid.num_images_fid // 1000}k ema  {ema}")
            mlflow.log_metric(metric_name, fid, step=global_step)
        # del model

    mlflow.end_run()


if __name__ == "__main__":
    log_fid()
