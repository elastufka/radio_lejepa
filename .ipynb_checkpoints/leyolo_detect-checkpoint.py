# train_yolo_full.py

import os
import sys
import torch.nn as nn
from torchvision.ops import nms
from torch.utils.data import DataLoader
#sys.path.append('/home/users/l/lastufka')
#from feuerzeug.datasets.COCODataset import MGCLSodDataset
import wandb
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
#from ultralytics.utils.ops.torch_utils import non_max_suppression
import cv2
import numpy as np
#from ultralytics.data.loaders import LoadImagesAndLabels
import argparse
from ultralytics import YOLO
import torch
import wandb
sys.path.append('/home/users/l/lastufka')
from feuerzeug.models import print_trainable_parameters
from ultralytics.models.yolo.detect.train import DetectionTrainer

class FrozenTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)

        # freeze backbone HERE (correct instance)
        for name, p in model.named_parameters():
            if "model.22" not in name:
                p.requires_grad = False

        print("[DEBUG] Backbone frozen inside Trainer")

        return model
def add_wandb_logging(model):
    def on_fit_epoch_end(trainer):
        metrics = trainer.metrics
        log_dict = {}
        
        for i, name in enumerate(trainer.loss_names):
            log_dict[f"train/{name}"] = float(trainer.tloss[i])
        
        for k, v in metrics.items():
            log_dict[k] = v
        
        wandb.log(log_dict)

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
def normalize_yolo_state_dict(sd):
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone"):
            k = k.replace("backbone.model.", "") 
        else:
            k = k.replace("model.", "")  # safest for YOLO v8 exports
        new_sd[k] = v
    return new_sd

def check_weights(jepa_ckpt, model):
    ckpt = normalize_yolo_state_dict(torch.load(jepa_ckpt, map_location="cpu")["state_dict"])
    yolo_sd = model.model.model.state_dict()    
    diff = 0
    for k in yolo_sd:
        if k in ckpt:
            if not torch.allclose(yolo_sd[k].cpu(), ckpt[k], atol=1e-6):
                diff += 1
                print("Mismatched tensors:", diff)
        else:
            print(f"{k} not in ckpt!")

def load_jepa_into_yolo(model, jepa_ckpt_path, backbone_end=10):
    """
    Load JEPA pretrained backbone weights into YOLO model.
    """

    ckpt = torch.load(jepa_ckpt_path, map_location="cpu")
    jepa_state = normalize_yolo_state_dict(ckpt["state_dict"])
    #backbone = model.model.model[:backbone_end]
    #backbone_modules = list(model.model.model[:backbone_end])
    backbone = model.model.model #nn.Sequential(*backbone_modules)
    
    #missing, unexpected = model.model.model[:backbone_end].load_state_dict(jepa_state, strict=False)
    missing, unexpected = backbone.load_state_dict(jepa_state, strict=False)
    #model.model.model[:backbone_end] = backbone
    print("[INFO] JEPA backbone loaded into YOLO")
    #print(missing)
    #print(unexpected)
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    return model

def hard_freeze(trainer):
    model = trainer.model

    for name, p in model.named_parameters():
        if "model.22" not in name:
            p.requires_grad = False

    print("[DEBUG] HARD FREEZE applied")

def build_test_loader(data_yaml, imgsz=640, batch=16):
    data = check_det_dataset(data_yaml)
    dataset = YOLODataset(
        img_path=data["val"],
        data=data,
        task="detect",
        imgsz=imgsz,
        augment=False,
        rect=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )

def on_train_start(trainer):
    print("INSIDE:", sum(p.mean().item() for p in trainer.model.parameters()))

def make_compare_to_jepa_callback(jepa_path):

    def compare_to_jepa(trainer):
        current = normalize_yolo_state_dict(trainer.model.state_dict())

        ckpt = torch.load(jepa_path, map_location="cpu")
        ref = normalize_yolo_state_dict(ckpt["state_dict"])
        #print(current.keys())
        #print("\n")
        #print(ref.keys())

        keys = ["0.conv.weight","4.cv1.conv.weight","8.cv2.conv.weight",]
        for key in keys:
            diff = (current[key].cpu() - ref[key]).abs().mean()
            print(f"\n[DEBUG] {key}")
            print(f"mean abs diff from JEPA ckpt = {diff:.10f}\n")

    return compare_to_jepa

def log_trainable_params(trainer):
    total_params = 0
    trainable_params = 0

    for p in trainer.model.parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n

    print("\n[DEBUG] ===== TRAINABLE PARAMETER CHECK =====")
    print(f"trainable params: {trainable_params:,}")
    print(f"all params: {total_params:,}")
    print(f"trainable%: {100 * trainable_params / total_params:.2f}")
    print("===========================================\n")

def yolo_decode_nms(pred, conf=0.5, iou=0.5):
    results = []

    for p in pred:
        if p is None or len(p) == 0:
            results.append(None)
            continue

        # p: [N, 6] = x1,y1,x2,y2,conf,cls
        scores = p[:, 4]
        keep = scores > conf
        p = p[keep]

        if len(p) == 0:
            results.append(None)
            continue

        boxes = p[:, :4]
        scores = p[:, 4]
        cls = p[:, 5]

        final_out = []

        for c in cls.unique():
            mask = cls == c
            idx = nms(boxes[mask], scores[mask], iou)

            selected = p[mask][idx]
            final_out.append(selected)

        if len(final_out):
            results.append(torch.cat(final_out, dim=0))
        else:
            results.append(None)

    return results

def train(
    model_name,
    data,
    jepa_ckpt,
    epochs,
    imgsz,
    batch,
    lr0,
    freeze_mode,
    freeze_epochs,
    backbone_end,
    project
):

    model = YOLO(model_name)

    # load JEPA weights if provided
    if jepa_ckpt is not None:
        model = load_jepa_into_yolo(model, jepa_ckpt, backbone_end)
        check_weights(jepa_ckpt, model)
        print(f"Weights loading from {jepa_ckpt} checked")
        model.add_callback("on_train_start",make_compare_to_jepa_callback(jepa_ckpt))
    #add_wandb_logging(model, log_jepa=(jepa_ckpt is not None))
    add_wandb_logging(model)
    # freeze policy
    if freeze_mode == "static":
        model.add_callback("on_train_start", hard_freeze)
        # #model = freeze_yolo_backbone(model, backbone_end)
        # model.model.requires_grad_(False)
        #print("")
        # # unfreeze head ONLY
        # for p in model.model.model[-1].parameters():
        #     p.requires_grad = True

    elif freeze_mode == "schedule":
        model = freeze_yolo_backbone(model, backbone_end)

    elif freeze_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown freeze_mode: {freeze_mode}")
        
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_fit_epoch_end", log_trainable_params)
    print_trainable_parameters(model)
    print("OUTSIDE:", sum(p.mean().item() for p in model.model.parameters()))
    # epoch-wise training control
    # for epoch in range(epochs):

    #     if freeze_mode == "schedule" and epoch == freeze_epochs:
    #         model = unfreeze_yolo_backbone(model, backbone_end)
    #         lr0 = lr0 * 0.1  # stabilize fine-tune transition

    #     print(f"\n[Epoch {epoch}/{epochs}] Training...")

    model.train(
        trainer=FrozenTrainer,
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        optimizer="AdamW",
        pretrained=True,
        verbose=True,
        #logger="wandb",
        project=project,
        freeze=None
        #name=run_name
    )

    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--data", type=str, default="/home/users/l/lastufka/scratch/rgz/od/dataset.yaml")
    parser.add_argument("--jepa_ckpt", type=str, default="/home/users/l/lastufka/scratch/lejepa/yolo8n_rgz/checkpoint_9.pt")
    parser.add_argument("--output_dir", type=str, default="/home/users/l/lastufka/scratch/yolo")
    parser.add_argument("--backbone_end", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--run_name", type=str, default="rgz_yolo")
    parser.add_argument(
        "--freeze_mode",
        type=str,
        default="schedule",
        choices=["none", "static", "schedule"],
    )
    parser.add_argument("--freeze_epochs", type=int, default=20)
    return parser.parse_args()


# -----------------------------
# main
# -----------------------------

def unpack_batch(batch):
    # Case 1: tuple
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]

    # Case 2: dict-style (Ultralytics newer format)
    if isinstance(batch, dict):
        imgs = batch.get("img", None)

        # YOLO format GT boxes often live here:
        targets = batch.get("bboxes", None)
        if targets is None:
            targets = batch.get("labels", None)

        return imgs, targets

    # Case 3: ultralytics "Batch" object (attribute-based)
    if hasattr(batch, "img"):
        imgs = batch.img
        targets = getattr(batch, "bboxes", None) or getattr(batch, "labels", None)
        return imgs, targets

    raise TypeError(f"Unknown batch type: {type(batch)}")

def yolo_xywhn_to_xyxy(box, img_h, img_w):
    """
    Convert normalized YOLO box (xc, yc, w, h)
    -> pixel xyxy (x1, y1, x2, y2)
    """
    xc, yc, bw, bh = box

    x1 = (xc - bw / 2) * img_w
    y1 = (yc - bh / 2) * img_h
    x2 = (xc + bw / 2) * img_w
    y2 = (yc + bh / 2) * img_h

    return x1, y1, x2, y2

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


def evaluate_image(preds, gts, iou_thresh=0.5, conf_thresh=0.5):
    """
    preds: list of (x1,y1,x2,y2,cls,score)
    gts: list of (x1,y1,x2,y2,cls)
    """

    preds = [p for p in preds if p[5] >= conf_thresh]

    matched_gt = set()
    matched_pred = set()

    # match predictions to GT
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if gi in matched_gt:
                continue

            if p[4] == g[4] and iou(p, g) >= iou_thresh:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    any_det = len(matched_gt) > 0
    all_det = len(matched_gt) == len(gts) if len(gts) > 0 else False
    perfect = (len(matched_gt) == len(gts)) and (len(preds) == len(gts))

    return int(any_det), int(all_det), int(perfect)

def compute_image_metrics(model, dataloader, conf=0.5, iou_thresh=0.5, max_batches=None):
    model.eval()

    any_correct = 0
    all_correct = 0
    perfect = 0
    total = 0

    for bi, batch in enumerate(dataloader):
        if max_batches and bi >= max_batches:
            break

        imgs = batch["img"]

        # GT (flattened across batch)
        bboxes = batch["bboxes"]      # (N, 4) in xyxy or xywh (depends on loader)
        cls = batch["cls"]            # (N,)
        batch_idx = batch["batch_idx"]

        # run model
        #imgs = imgs.float().to(next(model.model.parameters()).device) / 255.0
        imgs = imgs.float().to(model.device) / 255.0
        imgs = imgs.contiguous()
        #pred = model.model(imgs)
        results = model.predict(imgs,conf=conf,iou=iou_thresh,verbose=False)
        img_h, img_w = imgs.shape[2:]
        # group GT per image
        gt_per_img = {i: [] for i in range(len(imgs))}
        for i in range(len(bboxes)):
            c = int(cls[i])    
            x1, y1, x2, y2 = yolo_xywhn_to_xyxy(bboxes[i].tolist(),img_h,img_w)   
            gt_per_img[int(batch_idx[i])].append((x1, y1, x2, y2, c))

        # evaluate per image
        for i, r in enumerate(results):

            preds = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    c = int(b.cls[0])
                    s = float(b.conf[0])
                    preds.append((x1, y1, x2, y2, c, s))

            gt = gt_per_img.get(i, [])
            if i==0:
                print(gt[:3])
                print(preds[:3])
            any_det, all_det, perf = evaluate_image(preds, gt)

            any_correct += any_det
            all_correct += all_det
            perfect += perf
            total += 1

    return {
        "img/at_least_one": any_correct / total,
        "img/all_gt_detected": all_correct / total,
        "img/perfect": perfect / total,
    }

def main():
    args = get_args()
    print("\n========== CONFIG ==========")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("============================\n")
    wandb.init(project="LeJEPA-YOLO")
    model = train(
        model_name=args.model,
        data=args.data,
        jepa_ckpt=args.jepa_ckpt,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        freeze_mode=args.freeze_mode,
        freeze_epochs=args.freeze_epochs,
        backbone_end=args.backbone_end,
        project=args.output_dir
        #run_name=args.run_name
    )

    print("\n[INFO] Training complete")
    #return model
    test_loader = build_test_loader(args.data, imgsz=args.imgsz, batch=16)
    final_metrics = compute_image_metrics(
        model,
        test_loader,
        conf=0.5,
        iou_thresh=0.5
    )

    wandb.log(final_metrics)

if __name__ == "__main__":
    main()