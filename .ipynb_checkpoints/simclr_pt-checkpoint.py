import argparse
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import timm
import wandb
import shutil
import yaml
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
sys.path.append('/home/users/l/lastufka')
# import from sthalles repo
#from SimCLR.models.resnet_simclr import ResNetSimCLR  # we will replace backbone
from SimCLR.simclr import SimCLR
from feuerzeug.models import print_trainable_parameters
from feuerzeug.datasets.COCODataset import RGZimageDatasetClassification
from feuerzeug.datasets.PILDataset import RGZ20k
from feuerzeug.transforms import FakeChannels

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
# -----------------------------
# SimCLR Augmentations (paper-faithful)
# -----------------------------
def get_simclr_augmentation(image_size=224, fake_3chan = False):
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

# -----------------------------
# Dataset wrapper for SimCLR
# -----------------------------
class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

    def __len__(self):
        return len(self.dataset)


# -----------------------------
# ViT backbone wrapper
# -----------------------------
class ViTBackbone(nn.Module):
    def __init__(self, model_name="vit_tiny_patch16_224"):
        super().__init__()
        if "EUPE" in model_name:
            model_size = args.model_name[-1].lower()
            self.model = torch.hub.load(
            "/home/users/l/lastufka/EUPE",f"eupe_vit{model_size}16",source='local',weights=f"/home/users/l/lastufka/EUPE/EUPE-ViT-{model_size.upper()}.pt")
        else:
            self.model = timm.create_model(model_name, pretrained=False)

        # remove classification head
        try:
            self.model.reset_classifier(0)
        except AttributeError:
            if hasattr(self.model, "head"):
                self.model.head = torch.nn.Identity()
            elif hasattr(self.model, "classifier"):
                self.model.classifier = torch.nn.Identity()

        self.feature_dim = self.model.embed_dim

    def forward(self, x):
        return self.model(x)


# -----------------------------
# SimCLR Model
# -----------------------------
class SimCLRWithWandb(SimCLR):
    def train(self, train_loader, val_loader=None):
        scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16_precision)
        n_iter = 0

        for epoch in range(self.args.epochs):
            for (x_i, x_j) in train_loader:

                x_i = x_i.to(self.args.device)
                x_j = x_j.to(self.args.device)

                images = torch.cat([x_i, x_j], dim=0)

                with torch.cuda.amp.autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    wandb.log({
                        "loss": loss.item(),
                        "top1": top1[0],
                        "top5": top5[0],
                        "epoch": epoch,
                        "step": n_iter
                    })
                
                n_iter += 1

            if epoch >= 10:
                self.scheduler.step()

            print(f"Epoch {epoch} loss: {loss.item():.4f}")

            if val_loader is not None:
                val_loss = validation_loss(self, val_loader, args.device)
                wandb.log({"val_loss": val_loss})
           
            if self.args.output_dir:
                torch.save(self.model.state_dict(),f"{args.output_dir}/simclr_vit.pt")
                print("checkpoint saved")

@torch.no_grad()
def validation_loss(simclr_model, val_loader, device):
    simclr_model.model.eval()

    total_loss = 0.0
    n = 0

    for x_i, x_j in val_loader:
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        images = torch.cat([x_i, x_j], dim=0)

        features = simclr_model.model(images)
        logits, labels = simclr_model.info_nce_loss(features)
        loss = simclr_model.criterion(logits, labels)

        total_loss += loss.item()
        n += 1

    simclr_model.model.train()
    return total_loss / n

# -----------------------------
# Train loop
# -----------------------------
def train(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_views=2
    args.log_every_n_steps=100
    wandb.init(project=args.wandb_project, config=vars(args))

    transform = get_simclr_augmentation(args.image_size, fake_3chan = args.fake_chan)
    # dataset
    if args.dataset == "RGZ":
        ds = RGZimageDatasetClassification(None, None, train=True)
        test_ds = RGZimageDatasetClassification(None, None, train=False)
        val_transform = transform
    elif args.dataset == "RGZ20k":
        ds = RGZ20k(root = "/home/users/l/lastufka/scratch/rgz", train=True) 
        test_ds = RGZimageDatasetClassification(None, None, train=False)
        val_transform = get_simclr_augmentation(args.image_size)

    dataset = SimCLRDataset(ds, transform)
    val_dataset = SimCLRDataset(test_ds, val_transform)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        drop_last=True)
    val_loader = DataLoader(val_dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers,
                        drop_last=True)

    # model
    backbone = ViTBackbone(model_name= args.model_name)
    model = SimCLRWithWandb(
        args=args,
        model=backbone,
        optimizer=None,   # you pass real ones below
        scheduler=None,
        log_dir = args.output_dir
    )

    #criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)

    model.optimizer = optimizer
    model.scheduler = scheduler
    model.train(loader, val_loader=val_loader)

    if args.output_dir:
        torch.save(model.model.state_dict(),f"{args.output_dir}/simclr_vit.pt")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="RGZ")
    parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--fp16_precision", action="store_true")
    parser.add_argument("--fake_chan", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--wandb_project", type=str, default="simclr-vit")
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    train(args)