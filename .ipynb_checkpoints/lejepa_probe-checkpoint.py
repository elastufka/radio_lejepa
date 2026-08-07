import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
import timm, wandb, tqdm, hydra
from omegaconf import DictConfig
from datasets import load_dataset
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from transformers import Trainer, TrainingArguments
import evaluate
from PIL import Image
import numpy as np
import sys
import os
import argparse
sys.path.append("/home/users/l/lastufka")
#from feuerzeug.datasets.COCODataset import RGZimageDatasetClassification
#from feuerzeug.datasets.PILDataset import RGZ20k
from feuerzeug.datasets.NumPyDataset import Galaxy10Dataset
#from feuerzeug.datasets.MeerKATDataset import MeerKATDataset
from feuerzeug.models import ViTForLeJEPAProbe
from feuerzeug.transforms import CVStandardTransforms, CVEvalTransforms
#from galaxy_mnist.galaxy_mnist import GalaxyMNIST
from feuerzeug.hf_utils import *
from sklearn.metrics import f1_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit_small_patch16_224")
    parser.add_argument("--dataset_train", type=str, default="/home/users/l/lastufka/scratch/rgz")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--nlabels", type=int, default=100)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="./probe")
    parser.add_argument("--run_name", type=str, default="linear_probe")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--return_dict", action="store_true")
    parser.add_argument('--fake_chan', action = 'store_true',help='')
    #parser.add_argument('--bfloat16', action = 'store_true',help='')
    parser.add_argument('--center_crop', default=0,type=int,help='')
    parser.add_argument('--close_crop', action = 'store_true',help='')
    parser.add_argument('--flip', action = 'store_true',help='')
    parser.add_argument('--rotate', action = 'store_true',help='')
    parser.add_argument('--jitter', action = 'store_true',help='')
    return parser.parse_args()

def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {
        "Accuracy": float(accuracy),
        "F1": float(f1_micro),
        "macroF1": float(f1_macro),
    }

# def collate_fn(batch):
#     images, labels = zip(*batch)
#     return {
#         "pixel_values": torch.stack(images),
#         "labels": torch.tensor(labels)
#     }

def main(args):
    # train_ds = Galaxy10Dataset('/home/users/l/lastufka/scratch/Galaxy10DECals', train = True, transform = CVStandardTransforms(), return_dict = True)
    # val_ds = Galaxy10Dataset('/home/users/l/lastufka/scratch/Galaxy10DECals', train = False, transform = CVEvalTransforms(), return_dict=True)
    train_ds, val_ds, labels, loss_weights = get_dataset(args)
    model_size = args.model_name[args.model_name.find("vit")+3:args.model_name.find("vit")+4]
    size = "tiny" if model_size == 't' else "small"
    bbmodel = timm.create_model(f"vit_{size}_patch16_224", pretrained=False, num_classes=512,drop_path_rate=0.1,img_size=args.size,)
    #model.head = nn.Linear(model.embed_dim, len(labels))
    ckpt = torch.load(args.model_name, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    #state_dict.pop("head.weight", None)
    #state_dict.pop("head.bias", None)
    bbmodel.load_state_dict(state_dict, strict=False)
    #model = TimmWrapper(model)
    model = ViTForLeJEPAProbe(backbone_model=bbmodel,num_classes=args.num_classes,)

    if args.freeze_backbone:
        freeze_backbone(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        remove_unused_columns=False,
        seed=args.seed,
        label_names = ["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        data_collator=collate_fn
    )

    trainer.train()

    
if __name__ == "__main__":
    args = get_args()
    main(args)