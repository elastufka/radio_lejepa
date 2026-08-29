import argparse
import sys
import os
from transformers import ViTImageProcessor, Trainer, TrainingArguments,  ResNetForImageClassification,EarlyStoppingCallback, ConvNextImageProcessor
import evaluate  
import torch
from torch.optim import AdamW, SGD
from torchvision import models as tvmodels
import numpy as np
import timm

from feuerzeug.models import freeze_ViT_blocks, print_trainable_parameters, EUPEforClassification, CustomResNetClassifier, TimmWrapper,DinoClassifier

from feuerzeug.hf_utils import *
from peft import initialize_lora_eva_weights
from sklearn.metrics import precision_score, recall_score, f1_score
from safetensors.torch import load_file

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="nvidia/RADIO-B", type=str,help='')
    parser.add_argument('--output_dir', default="~/FM_compare/GMNIST", type=str,help='')
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
    parser.add_argument('--imsize', default=256, type=int,help='')
    parser.add_argument('--size', default=224, type=int,help='')
    parser.add_argument('--patience', default=5, type=int,help='')
    parser.add_argument('--thresh', default=0.5, type=float,help='')
    parser.add_argument('--lr', default=0.00005, type=float,help='')
    parser.add_argument('--lr_weight_backbone', default=0.1, type=float,help='')
    parser.add_argument('--lr_weight_class', default=10, type=float,help='')
    parser.add_argument('--weight_decay', default=0.01, type=float,help='')
    parser.add_argument('--resume', action = 'store_true',help='')
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
    parser.add_argument('--random', action = 'store_true',help='')
    parser.add_argument('--rho', default=2.0, type=float,help='')
    parser.add_argument('--freeze_backbone', default=0, type=int) #action = 'store_true',help='')
    parser.add_argument('--freeze_epochs', default=0, type=int) #action = 'store_true',help='')
    parser.add_argument('--use_fp16', dest="use_fp16",action = 'store_true',help='')
    parser.set_defaults(use_fp16=False)
    parser.add_argument('--img_fmt', default='npy', type=str,help='')
    parser.add_argument('--f1', default='micro', type=str,help='')
    parser.add_argument('--block', default=None, type=str,help='')
    parser.add_argument('--nblocks', default=1, type=int)
    parser.add_argument('--run_name', default=None, type=str,help='')
    parser.add_argument('--dataset_train', default='~/scratch/GalaxyMNIST') #MGCLS_data/enhanced/test_data_prep_cs/train', help='')
    parser.add_argument('--metadata', default=None)
    #parser.add_argument('--cuda', default=['0'], nargs='+', help='')
    args = parser.parse_args()
    return args

class SSLBackboneWrapper(nn.Module):
    """
    HF-compatible wrapper for SSL backbones (SimCLR/BYOL/DINO).
    Accepts pixel_values from Trainer.
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError("Backbone must expose num_features")

        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        """
        HuggingFace Trainer expects:
        - pixel_values (main input)
        - labels (optional)
        """
        #print("MODEL INPUT:", pixel_values.shape)
        x = pixel_values
        feats = self.backbone(x)
        logits = self.head(feats)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}

def _strip_ssl_state_dict(state_dict):
    """
    Handles:
    - module.
    - online_encoder.
    - target_encoder.
    - student / teacher
    """
    cleaned = {}

    for k, v in state_dict.items():
        k = k.replace("module.", "")
        k = k.replace("model.", "")

        # common SSL wrappers
        for prefix in [
            "online_encoder.",
            "target_encoder.",
            "student.",
            "teacher.",
            "backbone.",
            "encoder."
        ]:
            if k.startswith(prefix):
                k = k[len(prefix):]

        cleaned[k] = v

    return cleaned


def load_ssl_backbone_model(
    weights_path,
    num_classes,
    model_name="vit_tiny_patch16_224",
    device="cpu"
):
    """
    Drop-in loader for SimCLR / BYOL / DINOv1 checkpoints.

    Returns:
        nn.Module with .backbone + .classifier
    """

    # 1. Build backbone (IMPORTANT: num_classes=0 for feature extractor)
    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0
    )

    # 2. Load checkpoint
    ckpt = torch.load(weights_path, map_location=device)

    # 3. Extract state dict
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "online_network_state_dict" in ckpt:
            state_dict = ckpt["online_network_state_dict"]
        elif "target_network_state_dict" in ckpt:
            state_dict = ckpt["target_network_state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "online_encoder" in ckpt:
            state_dict = ckpt["online_encoder"]
        elif "teacher" in ckpt:
            state_dict = ckpt["teacher"]
        elif "student" in ckpt:
            state_dict = ckpt["student"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 4. Clean keys
    state_dict = _strip_ssl_state_dict(state_dict)

    # 5. Load weights
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[SSL Loader] Missing keys: {missing} | Unexpected keys: {unexpected}")

    # 6. Wrap for classification
    model = SSLBackboneWrapper(backbone, num_classes)
    return model

def get_optimizers(args,model):
    optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": args.lr*args.lr_weight_backbone},
    {"params": model.vit.parameters(), "lr": args.lr},
    {"params": model.classifier.parameters(), "lr": args.lr*args.lr_weight_class},
], weight_decay=args.weight_decay)
    return optimizer

def freeze_resnet_layers(model, nlayers=2, unfreeze_first_block = False):
    try:
        children = model.resnet.encoder.children()
    except AttributeError:
        children = model.backbone.children()

    for i, child in enumerate(children):
        print(i, child)
        if i == 0:
            continue
        elif i == 1 and unfreeze_first_block:
            continue
        elif i > nlayers:
            break
        else:
            for name, param in child.named_parameters():
                if not name.startswith('classifier'):
                    param.requires_grad = False
                    print(f"{name} frozen!")

    print(f"Model layers 1 - {nlayers} frozen")
    return model

class CustomTrainerRGZ(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 4.57664234, 3., 3.27130435, 3.38309353,2.92990654], device = logits.device)) #class weights
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class CustomTrainerFIRST(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.0861,1.,2.8316,3.3226], device = logits.device)) #class weights
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def transform(example_batch):
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs

def get_model(args, labels, loss_weights = None):
    print(args.model_name)
    state_dict = None
    if args.weights:
        state_dict = torch.load(args.weights)["model_state_dict"]
    print(args.freeze_backbone, not args.freeze_backbone,"state_dict", state_dict)

    if not "resnet" in args.model_name:
        lora_targets = ["key","query", "value","mlp","projection"]
        lkey="layer."
        if "dinov3" in args.model_name:
            model = DinoClassifier(args.model_name,num_classes=len(labels))#, state_dict = state_dict)
        elif "simclr" in args.model_name.lower()  or "byol" in args.model_name.lower(): #  or "dino" in args.model_name.lower():
            model = load_ssl_backbone_model(weights_path=args.model_name,num_classes=len(labels),model_name="vit_tiny_patch16_224", device="cpu")
        elif "efficientnet" in args.model_name:
            model = load_ssl_backbone_model(weights_path=args.model_name,num_classes=len(labels),model_name="efficientnet_b0", device="cpu")
        elif "eupe" in args.model_name.lower():    
            if "checkpoint" in args.model_name:
                if not args.model_name.endswith(".pt"):
                    ckpt_path = os.path.join(args.model_name,"model.safetensors")
                else:
                    ckpt_path = args.model_name
                model_size = args.model_name.split("EUPE-ViT-")[-1][0].lower()
            else:
                model_size = args.model_name[-1].lower()
                ckpt_path = None
            cls = False if args.tokens else True
            # if os.path.exists(args.model_name): #load from checkpoint
            #     print(args.model_name[args.model_name.find('eupe')+4:args.model_name.find('/checkpoint')])
            #     eupe_model = f"eupe_vit{args.model_name[args.model_name.find('eupe')+4:args.model_name.find('/checkpoint')]}16"
            #     print(eupe_model)
            # else:
            #     eupe_model = args.model_name
            fname={"t":"tiny","s":"small","b":"base"}
            arch = f"vit{model_size}16" if "ViT" in args.model_name else f"convnext_{fname[model_size]}"
            w = f"ViT-{model_size.upper()}" if "ViT" in args.model_name else f"ConvNeXt-{model_size.upper()}"
            print(arch, w)
            emodel = torch.hub.load(
            "~/EUPE",f"eupe_{arch}",source='local',weights=f"~/EUPE/EUPE-{w}.pt")
            #print(emodel)
            if ckpt_path is not None:
                if ckpt_path.endswith(".pt"):
                    state_dict = torch.load(ckpt_path, map_location="cpu")
                else:
                    state_dict = load_file(ckpt_path) #ckpt = torch.load(ckpt_path,map_location="cpu",)
                student_state = {}
                for k, v in state_dict.items():
                    k = k.replace("module.", "")
                    if k.startswith("student."):
                        new_key = k.replace("student.","")
                        student_state[new_key] = v
        
                missing, unexpected = emodel.load_state_dict(student_state,strict=False,)
                print("missing keys:", missing)
                print("unexpected keys:", unexpected)
            
            lora_targets = ["qkv", "proj"]
            fdict = {"t":192,"s":384,"b":768} if "vit" in arch else {"t":768, "s":768}
            block = model_size if args.block is not None else None
            # if os.path.exists(args.model_name):
            #     model = EUPEforClassification(emodel, num_classes = len(labels), feature_dim = fdict[model_size], lora=args.lora, lora_r = args.r, lora_alpha = args.alpha)
            #     model.load_state_dict(load_file(args.model_name), strict=False)
            #     eupe_backbone = model.backbone
            #     model = EUPEforClassification(
            #         backbone=eupe_backbone,
            #         num_classes=len(labels),
            #         feature_dim=fdict[model_size],   # EUPE-Tiny dim
            #         lora=args.lora, lora_r = args.r, lora_alpha = args.alpha, vitt = args.vitt, vits = args.vits, both=args.both,
            #         cls = cls, block = args.block
            #     )
            # else:
            model = EUPEforClassification(emodel, num_classes = len(labels), feature_dim = fdict[model_size],  vitt=args.vitt, vits = args.vits, both = args.both, mlp = args.mlp, cls=cls, block = block, nblocks = args.nblocks, lora=args.lora, lora_r = args.r, lora_alpha = args.alpha, lora_targets = lora_targets, normalize_features=True, random_weights=args.random) #
            print(model)
        elif "lejepa" in args.model_name:
            model_size = args.model_name[args.model_name.find("vit")+3:args.model_name.find("vit")+4]
            size = "tiny" if model_size == 't' else "small"
            print(size)
            model = timm.create_model(f"vit_{size}_patch16_224", pretrained=False, num_classes=512,drop_path_rate=0.1,img_size=args.size,)
            #model.head = nn.Linear(model.embed_dim, len(labels))
            model.head = nn.Sequential(nn.BatchNorm1d(model.embed_dim),nn.Linear(model.embed_dim, len(labels)))
            ckpt = torch.load(args.model_name, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            model.load_state_dict(state_dict, strict=False)
            model = TimmWrapper(model)
    else:
        lora_targets = [""]

        if args.mlp == 0 and not args.block:
            head_type = "linear" 
        elif args.block:
            head_type = "resnet_block"
        else:
            head_type = "mlp"
        model = CustomResNetClassifier(head_type=head_type,num_classes=len(labels), mlp_layers = args.mlp)
        if "timm" in args.model_name: #PEFT LoRA doesn't work with default Resnet config, need to rename things
            tvmodel = tvmodels.__dict__["resnet50"](weights="DEFAULT") #default weights
            tvmodel.fc = torch.nn.Identity() #torch.nn.Linear(in_features = , out_features = len(labels))
            state_dict = tvmodel.state_dict()
            config = get_timm_config(labels)

            with torch.no_grad():
                from_model = timm.create_model(args.model_name, pretrained=True, num_classes = len(labels)).eval()
                from_model.load_state_dict(state_dict,strict=False)
                our_model = ResNetForImageClassification(config).eval()
                module_transfer = convert_resnet_to_pytorch.ModuleTransfer(src=from_model, dest=our_model)
                x = torch.randn((1, 3, 224, 224))
                module_transfer(x)

            assert torch.allclose(from_model(x), our_model(x).logits), "The model logits don't match the original one."

            model = our_model
            print(model.state_dict()['resnet.embedder.embedder.convolution.weight'][:10,0,0,0])
            processor = ConvNextImageProcessor(do_normalize = False, do_resize = False, rescale_factor = 1)
            lora_targets = ["classifier","convolution"]
            lora_save = ["classifier", "normalization"]
    
    if args.freeze_backbone !=0:
        if 'resnet' in args.model_name:
            model = freeze_resnet_layers(model, nlayers = args.freeze_backbone)
        elif "vit" in args.model_name: #timm models
            for name, p in model.named_parameters():
                if "head" not in name:
                    p.requires_grad = False
        else:
            model = freeze_ViT_blocks(model, nblocks = args.freeze_backbone, lkey = lkey)

    print_trainable_parameters(model)

    return model
    
def get_trainer(args, model, collate_fn, train_ds, test_ds, processor, optimizer = None):
    training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.ims_per_batch,
    eval_strategy="epoch",
    logging_strategy = "epoch",
    save_strategy="epoch",
    num_train_epochs=args.epochs,
    fp16=args.use_fp16,
    learning_rate=args.lr,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='wandb',
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    label_names = ["labels"],
    #include_inputs_for_metrics = True
    load_best_model_at_end=True,
    dataloader_num_workers = args.num_workers,
    weight_decay = args.weight_decay,
    gradient_accumulation_steps=4, 
    run_name = args.run_name,
    #dataloader_persistent_workers = True,
    dataloader_pin_memory = True,
    seed = args.seed,
    data_seed = args.seed,
    warmup_ratio = 0.1,
    save_safetensors=False, #for convnext
    #label_smoothing_factor=0.1,
    #max_grad_norm=1.0
    #gradient_checkpointing=True
    )
    
    metrics = evaluate.combine(["accuracy","precision","recall","f1","confusion_matrix"])
    #print("metrics",metrics)
    ametric = evaluate.load("accuracy", experiment_id = args.eid, cache_dir = None)
    #pmetric = evaluate.load("precision", experiment_id = args.eid)
    #rmetric = evaluate.load("recall", experiment_id = args.eid)
    #fmetric = evaluate.load("f1", experiment_id = args.eid)
    #cmetric = load_metric("confusion_matrix", experiment_id = args.eid)
    
    def compute_metrics(p):
        #print("computing metrics") #don't think this is ever called...
        acc = ametric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['accuracy']
        #pre = pmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average= "micro")['precision']
        pre = precision_score(
            p.label_ids,
            np.argmax(p.predictions, axis=1),
            average=args.f1
        )
        #rec = rmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average= "micro")['recall']
        rec = recall_score(
            p.label_ids,
            np.argmax(p.predictions, axis=1),
            average=args.f1
        )
        f1 = f1_score(
            p.label_ids,
            np.argmax(p.predictions, axis=1),
            average="micro"
        )
        macf1 = f1_score(
            p.label_ids,
            np.argmax(p.predictions, axis=1),
            average="macro"
        )
        #f1  = fmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids, average= "micro")['f1']
        #conf = cmetric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)['confusion_matrix']
        return {"Accuracy":acc,"Precision":pre,"Recall":rec,"F1":f1, "macroF1":macf1} #,"Confusion Matrix": conf}
    
    tclass = Trainer
    if "rgz" in args.dataset_train:
        tclass = CustomTrainerRGZ
    if "FIRST" in args.dataset_train:
        tclass = CustomTrainerFIRST
    if "MNIST" in args.dataset_train or "radio" in args.model_name.lower():
        tclass = Trainer
    print("tclass",tclass)
    #else:
    #if "radio" in args.model_name.lower():
    #    tclass = CustomTrainerRGZ #Trainer #tclass = CustomTrainerGZ10
    if optimizer is None:
        trainer = tclass(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            processing_class=processor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
    else:        
        trainer = tclass(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=train_ds,
                eval_dataset=test_ds,
                processing_class=processor,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
                optimizers=(optimizer, None),
            )

    #trainer.add_callback(EvaluateEveryEpochCallback(trainer)) 
    if args.freeze_epochs !=0:
        trainer.add_callback(UnfreezeBackboneCallback(unfreeze_epochs = args.freeze_epochs))
    return trainer

def get_cosine_sim(model, train_ds, device="cuda"):
    model = model.to(device)
    model.eval()
    
    embeddings = []
    labels = []
    loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn = collate_fn
    )
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["pixel_values"].to(device)
            print("image diff:", (images[0] - images[1]).abs().mean())
            print("image equal:", torch.equal(images[0], images[1]))
            target = batch["labels"]
            features = model.backbone.forward_features(images)
            cls = features["x_norm_clstoken"] #cls = features[:, 0]
            patch = features["x_norm_patchtokens"]
            print("patch sample difference:",(patch[0] - patch[1]).abs().mean())
            print("patch batch variation:",patch.mean(dim=1).std(dim=0).mean())
            embeddings.append(cls.detach().cpu().clone())
            labels.append(target)
    
def run_main(args):
    size = args.size #256 if "resnet" in args.model_name else 224
    if "resnet" in args.model_name:
        processor = ConvNextImageProcessor(do_normalize = False, size = size)
    else:
        processor = ViTImageProcessor(do_normalize = False, size = size)
    print(args.dataset_train)
    train_ds, test_ds, labels, loss_weights = get_dataset(args)
    print("DATASET TYPE:", type(train_ds), len(train_ds))

    model = get_model(args, labels, loss_weights)

    if args.eva: 
        dataloader = torch.utils.data.DataLoader(train_ds, batch_size=args.ims_per_batch, shuffle=True)
        initialize_lora_eva_weights(model, dataloader)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay)

    trainer = get_trainer(args, model, collate_fn, train_ds, test_ds, processor, optimizer = optimizer)
    train_results = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print("training finished!")

if __name__ == "__main__":
    args = get_args()
    run_main(args)
