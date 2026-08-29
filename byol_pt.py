import argparse
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.models as models
import timm
import wandb
import shutil
import yaml
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.append('~/PyTorch-BYOL')
from trainer import BYOLTrainer2
from feuerzeug.datasets.COCODataset import RGZimageDatasetClassification
from feuerzeug.datasets.PILDataset import RGZ20k
from feuerzeug.transforms import FakeChannels

def get_augmentations(image_size=224, fake_3chan = False):
    """
    Based on SimCLR paper:
    - RandomResizedCrop
    - ColorJitter (strong)
    - RandomGrayscale
    - GaussianBlur (probabilistic)
    - HorizontalFlip
    """
    tlist = [T.ToTensor()]
    if fake_3chan:
        tlist.append(FakeChannels())
    tlist.append(T.RandomResizedCrop(image_size, scale=(0.5, 1.0))) #change from default
    tlist.append(T.RandomHorizontalFlip(p=0.5))
    tlist.append(T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8))
    tlist.append(T.RandomGrayscale(p=0.2))
    tlist.append(T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5))
    #tlist.append(T.ToTensor())
    tlist.append(T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
    
    return T.Compose(tlist)

class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x1 = self.transform(x)
        x2 = self.transform(x)
        return (x1, x2), y

    def __len__(self):
        return len(self.dataset)

def build_vit():
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    vit.reset_classifier(0)
    return vit

def train(args):

    wandb.init(project=args.wandb_project, config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == "RGZ":
        ds = RGZimageDatasetClassification(None, None, train=True)
        #test_ds = RGZimageDatasetClassification(None, None, train=False)
    elif args.dataset == "RGZ20k":
        ds = RGZ20k(root = "~/rgz", train=True) 
        #test_ds = RGZimageDatasetClassification(None, None, train=False)
    train_dataset = SimCLRDataset(ds, get_augmentations(args.image_size, args.fake_chan))

    # BYOL repo expects networks separately
    online_network = build_vit().to(device)
    target_network = build_vit().to(device)

    predictor = torch.nn.Sequential(
        torch.nn.Linear(128, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128)
    ).to(device)

    projector = nn.Sequential(
        nn.Linear(192, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Linear(512, 128)
    ).to(device)

    projector_target = nn.Sequential(
        nn.Linear(192, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Linear(512, 128)
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(online_network.parameters()) + list(predictor.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    trainer = BYOLTrainer2(
        online_network=online_network,
        target_network=target_network,
        predictor=predictor,
        projector=projector,
        projector_target=projector_target,
        optimizer=optimizer,
        device=device,
        max_epochs=args.epochs,
        m=args.m,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_interval=1,
        output_dir = args.output_dir
    )

    trainer.train(train_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="RGZ")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--fp16_precision", action="store_true")
    parser.add_argument("--fake_chan", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--m", type=float, default=0.999)
    parser.add_argument("--wandb_project", type=str, default="byol-vit")
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    train(args)