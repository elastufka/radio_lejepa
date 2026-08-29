import torch
import numpy as np
import sys

from feuerzeug.hf_utils import extract_features, get_dataset
from feuerzeug.plotting import plot_dataset_sample
from FM_compare.finetune import get_model
import scipy.linalg
import argparse
import umap
import matplotlib.pyplot as plt
import pickle
import pandas as pd

@torch.no_grad()
def compute_mmd(X, Y):
    """
    GPU accelerated MMD.
    torchmetrics uses an RBF kernel by default.
    """
    return mmd_rbf(X,Y)

@torch.no_grad()
def compute_stats(features):
    """
    Mean and covariance of embeddings.
    """
    mu = features.mean(dim=0)
    centered = features - mu
    cov = (centered.T @ centered) / (features.shape[0] - 1)
    return mu, cov


@torch.no_grad()
def compute_fid(X, Y, eps=1e-6):
    """
    Feature-space Fréchet distance.
    X, Y:
        [N,D] and [M,D] embeddings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu1, sigma1 = compute_stats(X)
    mu2, sigma2 = compute_stats(Y)

    diff = mu1 - mu2

    # scipy sqrtm is more robust for matrix square roots
    sigma1_np = sigma1.cpu().numpy()
    sigma2_np = sigma2.cpu().numpy()

    covmean, _ = scipy.linalg.sqrtm(
        sigma1_np @ sigma2_np,
        disp=False
    )

    # numerical cleanup
    if not torch.isfinite(torch.tensor(covmean)).all():
        offset = eps * torch.eye(
            sigma1.shape[0],
            device="cpu"
        ).numpy()

        covmean = scipy.linalg.sqrtm(
            (sigma1_np + offset)
            @
            (sigma2_np + offset)
        )

    covmean = torch.tensor(covmean.real,device=device,dtype=torch.float32)

    fid = (diff @ diff
        +
        torch.trace(sigma1+ sigma2- 2 * covmean))

    return fid

def quantify_distribution_shift(X,Y):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)

    fid = compute_fid(X, Y)
    X = torch.nn.functional.normalize(X, dim=1)
    Y = torch.nn.functional.normalize(Y, dim=1)
    mmd = compute_mmd(X, Y)
    print(f"MMD: {mmd.item():.6f}")
    print(f"FID: {fid.item():.6f}")
    return {"MMD":mmd.item(),"FID":fid.item()}

def embedding_mean_shift(X, Y):
    """
    Measures shift in embedding centroids.
    """
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    return torch.norm(mu_x - mu_y, p=2).item()

def covariance_distance(X, Y):
    """
    Compares covariance structure shift.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)

    cov_x = (Xc.T @ Xc) / (X.shape[0] - 1)
    cov_y = (Yc.T @ Yc) / (Y.shape[0] - 1)

    return torch.norm(cov_x - cov_y, p='fro').item()


def pairwise_sq_dists(A, B):
    """
    Squared Euclidean distances.

    A: [N, D]
    B: [M, D]
    returns: [N, M]
    """
    return (
        (A ** 2).sum(dim=1, keepdim=True)
        + (B ** 2).sum(dim=1, keepdim=True).T
        - 2.0 * (A @ B.T)
    ).clamp(min=0)


def median_bandwidth(X, Y, n_samples=2000):
    """
    Median heuristic for RBF kernel bandwidth.
    """

    Z = torch.cat([X, Y], dim=0)

    if Z.shape[0] > n_samples:
        idx = torch.randperm(
            Z.shape[0],
            device=Z.device
        )[:n_samples]
        Z = Z[idx]

    dists = pairwise_sq_dists(Z, Z)

    # remove zeros from diagonal
    sigma = torch.sqrt(
        torch.median(dists[dists > 0])
    )

    return sigma


def rbf_kernel(A, B, sigmas):
    """
    Multi-scale RBF kernel.
    """
    dists = pairwise_sq_dists(A, B)
    K = 0
    for sigma in sigmas:
        K += torch.exp(
            -dists / (2 * sigma ** 2)
        )

    return K / len(sigmas)


@torch.no_grad()
def mmd_rbf(X,Y,sigmas=None,chunk_size=1024,):
    """
    Memory-efficient RBF MMD.

    X: [N, D]
    Y: [M, D]

    Uses:
      - median heuristic bandwidth
      - multi-scale RBF kernels
      - chunked kernel computation

    Returns:
      MMD^2
    """

    if sigmas is None:
        sigma = median_bandwidth(X, Y)
        sigmas = [sigma / 2,sigma,sigma * 2,]

    def kernel_mean(A, B):
        """
        Compute mean kernel value:
        E[k(A,B)]
        without storing full kernel matrix.
        """
        total = 0.0
        count = 0

        for i in range(0, A.shape[0], chunk_size):
            Ai = A[i:i + chunk_size]
            K = rbf_kernel(
                Ai,
                B,
                sigmas
            )
            total += K.sum()
            count += K.numel()
        return total / count

    k_xx = kernel_mean(X, X)
    k_yy = kernel_mean(Y, Y)
    k_xy = kernel_mean(X, Y)
    return k_xx + k_yy - 2 * k_xy

def cka_linear(X, Y):
    """
    Linear Centered Kernel Alignment.
    Measures representation similarity.
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic = torch.norm(X.T @ Y, 'fro')**2
    var_x = torch.norm(X.T @ X, 'fro')
    var_y = torch.norm(Y.T @ Y, 'fro')

    return (hsic / (var_x * var_y)).item()

def median_sigma(X):
    dists = torch.cdist(X, X)
    return torch.median(dists).item()

def convert2PIL():
    train_ds = FIRSTGalaxyData(root="~/scratch/FIRSTGalaxyData", selected_split="train", input_data_list=["galaxy_data_h5.h5"], transform=None, is_PIL=True, is_RGB=True)
    test_ds = FIRSTGalaxyData(root="~/scratch/FIRSTGalaxyData", selected_split="test", input_data_list=["galaxy_data_h5.h5"], transform=None, is_PIL=True, is_RGB=True)
    for i in range(len(train_ds)):
        img,label = train_ds[i]
        img.save(f"~/scratch/FIRSTGalaxyData/pngs/train{i}_{label}.png")
    for i in range(len(test_ds)):
        img,label = test_ds[i]
        img.save(f"~/scratch/FIRSTGalaxyData/pngs/test{i}_{label}.png")

def plot_umap(model, args):
    try:
        with open('~/embeddings.pkl', 'rb') as fp:
            embeddings= pickle.load(fp)
    except FileNotFoundError:
        args.dataset_train = "RGZ20k"
        dataset, _,_,_ = get_dataset(args)
        train_embeddings,_ = extract_features(model, dataset)
        args.dataset_train = "~/scratch/FIRSTGalaxyData/pngs"
        dataset, _,_,_ = get_dataset(args)
        first_embeddings,_ = extract_features(model, dataset)
        args.dataset_train = "image_net1k"
        dataset, _,_,_ = get_dataset(args)
        imagenet_embeddings,_ = extract_features(model, dataset, collate_fn = None)
        args.dataset_train = "~/scratch/MiraBest"
        args.fake_chan = True
        dataset, _,_,_ = get_dataset(args)
        mb_embeddings,_ = extract_features(model, dataset)
        args.dataset_train = "~/scratch/rgz/od"
        dataset, _,_,_ = get_dataset(args)
        fig = plot_dataset_sample(dataset)
        fig.savefig("~/rgzod_sample.png",dpi=150)
        rgz_embeddings,_ = extract_features(model, dataset)
        
        embeddings = {
            "RGZ20k": train_embeddings,
            "RGZ": rgz_embeddings,
            "FIRST": first_embeddings,
            "MB": mb_embeddings,
            "ImageNet": imagenet_embeddings,
        }
        #print([k,v.shape() for k,v in embeddings])
        with open('~/embeddings.pkl', 'wb') as fp:
            pickle.dump(embeddings, fp)

    features = []
    labels = []
    
    for name, emb in embeddings.items():
    
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()
        print(emb.shape)
        if len(emb) > 10000:
            emb = emb[:10000]
        features.append(emb)
        labels.extend([name] * len(emb))
    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    print("Embedding matrix:", features.shape)

    features = features / np.linalg.norm(features,axis=1,keepdims=True)
    reducer = umap.UMAP(n_neighbors=50,min_dist=0.1,metric="cosine",random_state=42,) 
    embedding_2d = reducer.fit_transform(features)

    plt.figure(figsize=(8, 6))
    unique_labels = ["RGZ20k","RGZ","ImageNet","FIRST","MB"]
    print(unique_labels)
    # Let matplotlib choose colors automatically
    for label in unique_labels:
        mask = labels == label
    
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=5,
            alpha=0.5,
            label=label,
            rasterized=True,  # important for large datasets
        )
    
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP of Feature Embeddings")
    plt.legend(markerscale=3,fontsize=10,frameon=True,)
    
    plt.tight_layout()
    plt.savefig("embedding_umap.png",dpi=300,bbox_inches="tight")
    plt.close()

def calc_all_metrics(args):
    with open(args.dataset_train, 'rb') as fp:
        embeddings= pickle.load(fp)
    keys = list(embeddings.keys())
    dd,ii = {},{}
    for i in range(len(keys)-1):
        if keys[i] != "RGZ20k":
            e0 = embeddings["RGZ20k"]
            e1 = embeddings[keys[i]]
            print(f"distribution shift between {keys[i]} and RGZ20k:")
            dd[f"RGZ20k - {keys[i]}"] = quantify_distribution_shift(e0,e1)
    for i in range(len(keys)-1):
        if keys[i] != "ImageNet":
            e0 = embeddings[keys[i]]
            e1 = embeddings["ImageNet"]
            print(f"distribution shift between {keys[i]} and ImageNet:")
            ii[f"ImageNet - {keys[i]}"] = quantify_distribution_shift(e0,e1)
    return dd,ii

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="nvidia/RADIO-B", type=str,help='')
    parser.add_argument('--output_dir', default="~/GMNIST", type=str,help='')
    parser.add_argument('--project', default="FM_compare", type=str,help='')
    parser.add_argument('--weights', default=None, type=str,help='')
    parser.add_argument('--ims_per_batch', default=8, type=int,help='')
    parser.add_argument('--nlabels', default=100, type=int,help='')
    parser.add_argument('--nlayers', default=1, type=int,help='')
    parser.add_argument('--fake_chan', action = 'store_true',help='')
    parser.add_argument('--bfloat16', action = 'store_true',help='')
    parser.add_argument('--center_crop', default=0,type=int,help='')
    parser.add_argument('--close_crop', action = 'store_true',help='')
    parser.add_argument('--flip', action = 'store_true',help='')
    parser.add_argument('--rotate', action = 'store_true',help='')
    parser.add_argument('--jitter', action = 'store_true',help='')
    parser.add_argument('--whitening', action = 'store_true',help='')
    parser.add_argument('--normalize', action = 'store_true',help='')
    parser.add_argument('--num_workers', default=4, type=int,help='')
    parser.add_argument('--eid', default=1, type=int,help='experimetn ID')
    parser.add_argument('--epochs', default=2, type=int,help='detectron2 default')
    parser.add_argument('--seed', default=15, type=int,help='')
    parser.add_argument('--size', default=224, type=int,help='')
    parser.add_argument('--vitt', action = 'store_true',help='')
    parser.add_argument('--vits', action = 'store_true',help='')
    parser.add_argument('--mlp', default=0, type=int,)
    parser.add_argument('--both', action = 'store_true',help='')
    parser.add_argument('--tokens', action = 'store_true',help='')
    parser.add_argument('--lora', action = 'store_true',help='')
    parser.add_argument('--r', default=16, type=int,help='')
    parser.add_argument('--alpha', default=16, type=int,help='')
    parser.add_argument('--lora_dropout', default=0.1, type=float,help='')
    parser.add_argument('--eva', action = 'store_true',help='')
    parser.add_argument('--cv', action = 'store_true',help='')
    parser.add_argument('--rho', default=2.0, type=float,help='')
    parser.add_argument('--use_fp16', dest="use_fp16",action = 'store_true',help='')
    parser.set_defaults(use_fp16=False)
    parser.add_argument('--dataset_train', default='~/scratch/image_net1k') 
    parser.add_argument('--dataset2', default='~/scratch/rgz/od') 
    parser.add_argument('--freeze_backbone', default=0, type=int) #action = 'store_true',help='')
    parser.add_argument('--block', default=None, type=str,help='')
    parser.add_argument('--nblocks', default=1, type=int)
    parser.add_argument('--random', action = 'store_true',help='')
    parser.add_argument('--return_dict', action = 'store_true',help='')
    parser.add_argument('--umap', action = 'store_true',help='')
    parser.add_argument('--metadata', default=None)
    parser.add_argument('--imsize', default=256, type=int,help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #convert2PIL()
    # load backbone
    args = get_args()
    if args.dataset_train.endswith("pkl"):
        dd, ii = calc_all_metrics(args)
        d0 = pd.DataFrame(dd)
        d1 = pd.DataFrame(ii)
        df = pd.concat([d0,d1])
        df.to_csv(f"~/{args.dataset_train[args.dataset_train.rfind('/'):-4]}_mets.csv")
        sys.exit()
    backbone = get_model(args, [])
    try:
        backbone = backbone.backbone
    except AttributeError:
        print(backbone)
        backbone = backbone.model
    if args.umap:
        plot_umap(backbone, args)
        sys.exit()
        
    print(args.dataset_train)
    dataset, _,_,_ = get_dataset(args)
    args.dataset_train = args.dataset2
    print(args.dataset_train)
    #args.fake_chan = True
    dataset2, _,_,_ = get_dataset(args)

    X,_ = extract_features(backbone, dataset, collate_fn=None)
    Y,_ = extract_features(backbone, dataset2)
    quantify_distribution_shift(X,Y)