import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
import sys
import random
from transformers import (
    Trainer,
    TrainingArguments,
    ViTImageProcessor,
)

import timm
sys.path.append("/home/users/l/lastufka")
from feuerzeug.hf_utils import *
from feuerzeug.transforms import *
#from feuerzeug.models import *
from feuerzeug.datasets.PILDataset import RGZ20k
from feuerzeug.datasets.NumPyDataset import Galaxy10Dataset
from feuerzeug.datasets.MeerKATDataset import MeerKATDataset
#from feuerzeug.datasets.NumPyDataset import Galaxy10Dataset, Galaxy10Subset, Galaxy10xmatch, FeatureDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_model',type=str,required=True,default="EUPE-ViT-S")
    parser.add_argument('--output_dir',type=str,default='./eupe_distill_output',)
    parser.add_argument('--dataset_train',type=str,required=True,default = "/home/users/l/lastufka/scratch/rgz")
    parser.add_argument('--dataset_eval',type=str,default=None,)
    parser.add_argument('--epochs',type=int,default=100,)
    parser.add_argument('--seed',type=int,default=14,)
    parser.add_argument('--ims_per_batch',type=int,default=64,)
    parser.add_argument('--lr',type=float,default=1e-4,)
    parser.add_argument('--weight_decay',type=float,default=0.05,)
    parser.add_argument('--num_workers',type=int,default=8,)
    parser.add_argument('--size',type=int,default=224,)
    parser.add_argument('--patience',type=int,default=10,)
    parser.add_argument('--bf16',action='store_true',)
    parser.add_argument('--fp16',action='store_true',)
    parser.add_argument('--run_name',type=str,default='eupe_distill',)
    parser.add_argument('--lambda_global',type=float,default=1.0,)
    parser.add_argument('--lambda_patch',type=float,default=1.0,)
    parser.add_argument('--lambda_hidden',type=float,default=0.5,)
    parser.add_argument('--eval_frac',type=float,default=0.2,)
    args = parser.parse_args()
    return args

def extract_features(model, images):

    hidden_states = []
    x = model.patch_embed(images)
    #print("PATCH SHAPE:", x.shape)
    # BHWC -> BNC
    if x.ndim == 4:
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)


    cls_token = model.cls_token.expand(
        x.shape[0], -1, -1
    )

    x = torch.cat((cls_token, x), dim=1)
    #x = x + model.pos_embed
    #x = model.pos_drop(x)

    for block in model.blocks:
        x = block(x)
        hidden_states.append(x)

    x = model.norm(x)

    cls_token = x[:, 0]
    patch_tokens = x[:, 1:]

    return cls_token, patch_tokens, hidden_states

def cosine_loss(x, y):

    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    return 1 - F.cosine_similarity(
        x,
        y,
        dim=-1,
    ).mean()

class ProjectionHead(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class EUPEViTDistillModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.lambda_global = args.lambda_global
        self.lambda_patch = args.lambda_patch
        self.lambda_hidden = args.lambda_hidden

        self.teacher = torch.hub.load(
            "/home/users/l/lastufka/EUPE",
            "eupe_vits16",
            source="local",
            weights=f"/home/users/l/lastufka/EUPE/{args.teacher_model}.pt",
        )

        self.teacher.eval()

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        self.projector = ProjectionHead(
            192,
            384,
        )

    def forward(self, pixel_values, **kwargs):

        with torch.no_grad():

            t_cls, t_patch, t_hidden = extract_features(
                self.teacher,
                pixel_values,
            )

        s_cls, s_patch, s_hidden = extract_features(
            self.student,
            pixel_values,
        )

        s_cls = self.projector(s_cls)

        s_patch = self.projector(s_patch)

        loss_global = cosine_loss(
            s_cls,
            t_cls,
        )

        loss_patch = cosine_loss(
            s_patch,
            t_patch,
        )

        loss_hidden = 0.0

        student_layers = len(s_hidden)
        teacher_layers = len(t_hidden)

        for i in range(student_layers):

            teacher_idx = int(
                i * teacher_layers / student_layers
            )

            s_feat = s_hidden[i]
            t_feat = t_hidden[teacher_idx]

            s_feat = self.projector(s_feat)

            loss_hidden += cosine_loss(
                s_feat,
                t_feat,
            )

        loss_hidden /= student_layers

        loss = (
            self.lambda_global * loss_global +
            self.lambda_patch * loss_patch +
            self.lambda_hidden * loss_hidden
        )

        return {
            "loss": loss,
            "loss_global": loss_global.detach(),
            "loss_patch": loss_patch.detach(),
            "loss_hidden": loss_hidden.detach(),
        }

def collate_fn(batch):

    pixel_values = torch.stack([
        x[0] for x in batch
    ])

    labels = None
    if isinstance(batch[0], (tuple, list)) and len(batch[0]) > 1:
        labels = torch.tensor([x[1] for x in batch])

    if labels is not None:
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

    return {
        "pixel_values": pixel_values
    }


# =========================================================
# TRAINER
# =========================================================

class DistillationTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        outputs = model(**inputs)
        loss = outputs["loss"]

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):

        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]

        return (
            loss.detach(),
            None,
            None,
        )

# =========================================================
# TRAINER SETUP
# =========================================================

def get_trainer(args, model, train_ds, eval_ds=None):

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.ims_per_batch,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy=(
            "epoch"
            if eval_ds is not None
            else "no"
        ),

        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_total_limit=2,
        report_to="wandb",
        run_name=args.run_name,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=args.patience
        )],
    )

    return trainer


# =========================================================
# MAIN
# =========================================================

def run_main(args):

    processor = ViTImageProcessor(
        do_normalize=False,
        size=args.size,
    )

    # -----------------------------------------------------
    # LOAD DATASETS
    # -----------------------------------------------------
    if "rgz" in args.dataset_train:
        ds = RGZ20k(root = args.dataset_train,transform=CVEvalTransforms(),)
    elif "Galaxy10" in args.dataset_train:
        ds = Galaxy10Dataset(args.dataset_train, train = True, transform = CVEvalTransforms())
    elif "MGCLS" in args.dataset_train:
        ds0 = RGZ20k(root = "/home/users/l/lastufka/scratch/rgz",transform=CVEvalTransforms(),)
        x = ds0[0]
        #print(x[0].shape)
        tlist = transforms.Compose([transforms.ToTensor(),transforms.Resize((args.size, args.size)),
            #    FakeChannels(),
            #transforms.Lambda(lambda x: x.repeat(3, 1, 1)if x.shape[0] == 1else x),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))]) 
        ds1 = MeerKATDataset(args.dataset_train, transform=tlist, fake_3chan = False) #take 20k of this...
        #print(ds1[0][0].shape)
        random.seed(args.seed)
        indices = random.sample(range(len(ds1)), 20_000)
        ds1_subset = Subset(ds1, indices)
        ds = ConcatDataset([ds0, ds1_subset])

        ds = ds1
        #ds = ds.shuffle(seed=args.seed)

    eval_size = int(len(ds) * args.eval_frac)
    train_size = len(ds) - eval_size
    
    train_ds, eval_ds = random_split(
        ds,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(len(train_ds),len(eval_ds))

    model = EUPEViTDistillModel(args)

    trainer = get_trainer(
        args,
        model,
        train_ds,
        eval_ds,
    )

    trainer.train()
    trainer.save_model()
    print("training complete")


if __name__ == "__main__":

    args = get_args()
    run_main(args)