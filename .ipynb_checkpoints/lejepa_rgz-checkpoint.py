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
from PIL import Image
import numpy as np
from ultralytics import YOLO
import sys
import os
sys.path.append("/home/users/l/lastufka")
from feuerzeug.datasets.COCODataset import RGZimageDatasetClassification
from feuerzeug.datasets.PILDataset import RGZ20k
from feuerzeug.datasets.NumPyDataset import Galaxy10Dataset
from feuerzeug.datasets.MeerKATDataset import MeerKATDataset
from feuerzeug.models import print_trainable_parameters
from galaxy_mnist.galaxy_mnist import GalaxyMNIST

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class EfficientNetEncoder(nn.Module):
    def __init__(
        self,
        model_name="efficientnet_b0",
        proj_dim=128,
        mlp_dim=1024,
        pretrained=True,
    ):
        super().__init__()
        if os.path.isfile(model_name):
            ckpt = torch.load(model_name, map_location="cpu")
            backbone_name = ckpt["model_name"]
        else:
            backbone_name = model_name

        gpool = "" if "efficient" in model_name else None
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,      # remove classifier
            global_pool=gpool       # return feature map
        )
        self.backbone.set_grad_checkpointing(True)
        
        if os.path.isfile(model_name):
            nsd = {k.replace("backbone.",""):v for k,v in ckpt["state_dict"].items() if "proj" not in k}
            self.backbone.load_state_dict(nsd)
        self.embed_dim = self.backbone.num_features

        self.proj = MLP(
            self.embed_dim,
            [mlp_dim, mlp_dim, proj_dim],
            norm_layer=None,
        )

    def forward(self, x):
        N, V = x.shape[:2]
        x = x.flatten(0, 1)
        #print("input:", x.shape)
        feat = self.backbone.forward_features(x)
        #print("feature:", feat.shape)
        #feat = self.backbone.forward_features(x)      # (B, C, H, W)
        emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)

        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)

        return emb, proj

class ViTEncoder(nn.Module):
    def __init__(self, model_name, proj_dim=128, imsize=128):
        super().__init__()
        if os.path.isfile(model_name):
            ckpt = torch.load(model_name, map_location="cpu")
            backbone_name = ckpt["model_name"]
        else:
            backbone_name = model_name
            
        fdict = {"t":192,"s":384,"b":768}
        if "EUPE" in model_name:
            model_size = model_name[-1].lower()
            
            self.backbone = torch.hub.load(
            "/home/users/l/lastufka/EUPE",f"eupe_vit{model_size}16",source='local',weights=f"/home/users/l/lastufka/EUPE/EUPE-ViT-{model_size.upper()}.pt")
            self.embed_dim = fdict[model_size]
        elif "dino" in model_name:
            model_size = model_name[8] #e.g. "dino_vits16"
            self.backbone = torch.hub.load("facebookresearch/dino:main",model_name,pretrained=True)
            self.embed_dim = fdict[model_size]
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=512,
                drop_path_rate=0.1,
                img_size=imsize,
            )
            self.embed_dim = 512

        if os.path.isfile(model_name):
            self.backbone.load_state_dict(ckpt["state_dict"])
        self.proj = MLP(self.embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)

class YOLOEncoder(nn.Module):
    def __init__(
        self,
        model_name="yolov8n.pt",
        proj_dim=128,
        mlp_dim=1024,
        imsize=224,
        use_spatial=False,
        C=1024
    ):
        super().__init__()

        yolo = YOLO(model_name)
        print_trainable_parameters(yolo)
        for p in yolo.model.parameters():
            p.requires_grad = True
        
        self.backbone = yolo.model #.to("cuda")
        self.backbone.model.training = True
        self.backbone.model[-1].training = False
        print_trainable_parameters(self.backbone)
        # backbone only
        
        self.use_spatial = use_spatial
        # self.feat = None

        def hook_fn(module, inp, out):
            self.feat = out #.detach()
            # print(type(self.feat))
            # print(self.feat.requires_grad)
            # print(self.feat.is_leaf)
            # print(self.feat.grad_fn)

        # register hook on last backbone block dynamically
        backbone_blocks = list(self.backbone.model.children())
        target = backbone_blocks[-3]   # typically last C2f before neck
        
        target.register_forward_hook(hook_fn)

        # with torch.no_grad():
        #     dummy = torch.zeros(1, 3, imsize, imsize).to("cuda")
        #     _ = self.backbone(dummy)
        
        # feat = self.feat
        # if isinstance(feat, (list, tuple)):
        #     feat = feat[-1]
        
        C = 384 #feat.shape[1]   # <-- THIS is correct YOLO channel dim
        self.embed_dim = C

        self.proj = MLP(C,[mlp_dim, mlp_dim, proj_dim],norm_layer=None) #nn.BatchNorm1d,)

    def forward(self, x):
        #print("grad enabled:", torch.is_grad_enabled())
        #print("autocast:", torch.is_autocast_enabled())
        N, V = x.shape[:2]
        x = x.flatten(0, 1)

        self.feat = None
        with torch.enable_grad():
            _ = self.backbone(x)   # full forward pass

        feat = self.feat 

        # run full backbone+neck, but stop BEFORE Detect head
        #y = self.backbone.forward(x)  # IMPORTANT: NOT self.model(x)
        #print("y requires_grad:", y.requires_grad if torch.is_tensor(y) else None)
        # YOLO returns tuple/list of feature maps in training path
        # if isinstance(y, (list, tuple)):
        #     feat = y[-1]   # highest-level feature map (P5)
    
        # elif isinstance(y, dict):
        #     feat = y.get("feat", None) or y.get("feats", None)
        #     if isinstance(feat, (list, tuple)):
        #         feat = feat[-1]
        #     if feat is None:
        #         feat = list(y.values())[-1]
    
        # else:
        #     feat = y
    
        # assert hasattr(feat, "shape"), f"Bad feat type: {type(feat)}"
        # print("feat grad fn:", feat.grad_fn)
        # print("feat requires_grad:", feat.requires_grad)
        emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
    
        return emb, proj
    
    # def forward(self, x):
    #     N, V = x.shape[:2]
    #     x = x.flatten(0, 1)
    
    #     # FORCE deterministic YOLO behavior (critical)
    #     self.backbone.model.train()
    
    #     y = self.backbone(x)
    
    #     # ---- robust feature extraction ----
    #     feat = None
    
    #     if isinstance(y, dict):
    #         feat = y.get("feat", None) or y.get("feats", None)
    #         if feat is None:
    #             feat = list(y.values())[-1]
    
    #     elif isinstance(y, (list, tuple)):
    #         feat = y[-1]
    
    #     else:
    #         feat = y
    
    #     # final safety unwrap
    #     if isinstance(feat, (list, tuple)):
    #         feat = feat[-1]
    
    #     assert hasattr(feat, "shape"), f"Invalid feat type: {type(feat)}"
    
    #     # ---- JEPA embedding ----
    #     emb = F.adaptive_avg_pool2d(feat, 1).flatten(1)
    
    #     proj = self.proj(emb)
    #     proj = proj.reshape(N, V, -1).transpose(0, 1)
    
    #     return emb, proj
    
    # def forward(self, x):
    #     N, V = x.shape[:2]
    #     x = x.flatten(0, 1)
    #     #_ = self.backbone(x)   # triggers hook
    #     #with torch.set_grad_enabled(self.training):
    #     y = self.backbone(x) #self.feat
    #     #print(type(y))
    #     if isinstance(y, dict):
    #         #print(y.keys())
    #         feat = y.get("feat", None) or y.get("feats", None)
    #         if feat is None:
    #             feat = list(y.values())[-1]
    #         if isinstance(feat, (list, tuple)):
    #             feat = feat[-1]
    #     elif isinstance(y, (list, tuple)):
    #         feat = y[-1]
    #         if isinstance(feat, dict):
    #             #print(feat.keys())
    #             feat = feat.get("feat", None) or feat.get("feats", None)
    #             #print(type(feat))
    #             #print(len(feat))
    #             feat = feat[-1]
    #             # fallback if keys missing
    #             if feat is None:
    #                 feat = list(feat.values())[-1]
    #     else:
    #         feat = y
    #     print(feat.shape)

    #     if self.use_spatial:
    #         emb = feat.flatten(2).mean(-1)
    #         proj_in = (feat.flatten(2).transpose(1, 2))
    #         proj = self.proj(proj_in.reshape(-1, proj_in.shape[-1]))
    #         proj = proj.reshape(N,V,-1,proj.shape[-1],)
    #         proj = proj.flatten(1, 2)
    #         proj = proj.transpose(0, 1)

    #     else:
    #         emb = F.adaptive_avg_pool2d(feat,1,).flatten(1)
    #         proj = self.proj(emb)
    #         proj = proj.reshape(N,V,-1,).transpose(0, 1)

    #     return emb, proj

class HFDataset(torch.utils.data.Dataset):
    def __init__(self, ds, split, V=1, V_local=2, imsize=224, local_imsize=98):
        self.V = V
        self.V_local = V_local
        self.ds = ds
        self.global_aug = v2.Compose(
            [
                v2.RandomResizedCrop(imsize, scale=(0.5, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                #v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                #v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.local_aug = v2.Compose(
            [
                v2.RandomResizedCrop(local_imsize,scale=(0.4, 0.7),),
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)],p=0.8,),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomHorizontalFlip(),

                # resize back to ViT input size
                v2.Resize((imsize, imsize)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
            ]
        )
        self.test = v2.Compose(
            [
                v2.Resize(imsize),
                v2.CenterCrop(imsize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, i):
        item = self.ds[i]
        #print(item.keys())
        if isinstance(item, dict):
            img = item["image"]
            label = item["id_str"]
            #print(type(img))
        else:
            img = item[0]
            label = item[1]
        if isinstance(img, np.ndarray):
            if img.shape[0] == 3:  # CHW → HWC
                img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img.astype(np.uint8))
        img = img.convert("RGB")

        # evaluation mode
        if self.V == 1 and self.V_local == 0:
            return self.test(img).unsqueeze(0), label
            
        #transform = self.aug if self.V > 1 else self.test
        #return torch.stack([transform(img) for _ in range(self.V)]), label

        global_views = torch.stack([self.global_aug(img) for _ in range(self.V)])
        if self.V_local != 0:
            local_views = torch.stack([self.local_aug(img) for _ in range(self.V_local)])
            views = torch.cat([global_views, local_views], dim=0)
        else:
            views = global_views

        # concatenate on V dimension
        
        return views, label

    def __len__(self):
        return len(self.ds)

class AddV:
    def __init__(self, V=1, V_local=0):
        self.V = V
        self.V_local = V_local
    def __call__(self, x):
        return torch.stack([x for _ in range(self.V)])

class MetricMeter:
    def __init__(self):
        self.sigreg = 0.0
        self.inv = 0.0
        self.probe = 0.0
        self.lejepa = 0.0
        self.n = 0

    def update(self, sigreg, inv, lejepa, bs, probe=0):
        self.sigreg += sigreg * bs
        self.inv += inv * bs
        self.probe += probe * bs
        self.lejepa += lejepa * bs
        self.n += bs

    def compute(self):
        return {
            "sigreg": self.sigreg / self.n,
            "inv": self.inv / self.n,
            "probe": self.probe / self.n,
            "lejepa": self.lejepa / self.n,
        }
        
@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if dist.is_initialized():
        is_main = dist.get_rank() == 0
    else:
        is_main = True
    if is_main:
        wandb.init(project="LeJEPA", name = cfg.run_name, config = dict(cfg))
        
    torch.manual_seed(cfg.seed)
    do_probe = False
    eval_transform = v2.Compose(
        [
            v2.Resize(cfg.imsize),
            v2.CenterCrop(cfg.imsize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AddV(V=1)
        ]
    )
    if cfg.dataset == "RGZ":
        ds = RGZimageDatasetClassification(None, None, train=True)
        test_ds = RGZimageDatasetClassification(None, None, train=False, transform = eval_transform)
        do_probe = True
    elif cfg.dataset == "RGZ20k":
        ds = RGZ20k(root = "/home/users/l/lastufka/scratch/rgz", train=True) 
        test_ds = RGZimageDatasetClassification(None, None, train=False, transform = eval_transform)
    elif cfg.dataset == "MGCLS":
        ds = MeerKATDataset('/home/users/l/lastufka/scratch/MGCLS_data/enhanced/crops_224_3chan_rescale', fake_3chan = False)
        test_ds = RGZimageDatasetClassification(None, None, train=False, transform = eval_transform)
    elif cfg.dataset == "GZ10":
        ds = Galaxy10Dataset('/home/users/l/lastufka/scratch/Galaxy10DECals', train = True)
        test_ds = Galaxy10Dataset('/home/users/l/lastufka/scratch/Galaxy10DECals', train = False, transform = eval_transform)
        do_probe = True
    elif cfg.dataset == "GMNIST":
        ds = GalaxyMNIST(root='/home/users/l/lastufka/scratch/GalaxyMNIST', download = False, train=True)
        test_ds = GalaxyMNIST(root='/home/users/l/lastufka/scratch/GalaxyMNIST', download = False, train=False, transform = eval_transform)
        do_probe = True
    elif cfg.dataset == "GZ2":
        ds = load_dataset('mwalmsley/gz2', split='train') 
        test_ds = GalaxyMNIST(root='/home/users/l/lastufka/scratch/GalaxyMNIST', download = False, train=False, transform = eval_transform) #load_dataset('mwalmsley/gz2', split='train') 
    full_ds = HFDataset(ds, "train", V=cfg.V, V_local=cfg.V_local, imsize=cfg.imsize)

    n_val = int(0.2 * len(full_ds))
    n_train = len(full_ds) - n_val
    
    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    #test_ds = HFDataset("validation", V=1)

    #print(test_ds[0][0].shape, test_ds[0][1])
    train = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True, num_workers=8)
    val = DataLoader(val_ds, batch_size=cfg.bs, shuffle=False, num_workers=8)
    test = DataLoader(test_ds, batch_size=cfg.bs, num_workers=8)
    print("data loaded")
    
    # modules and loss
    if cfg.checkpoint is None:
        mname = cfg.model
    else:
        mname = cfg.checkpoint
    if "yolo" in mname:
        net = YOLOEncoder(mname,proj_dim=cfg.proj_dim,mlp_dim=cfg.mlp_dim,use_spatial=cfg.yolo_spatial,).to("cuda")
    elif "efficient" in mname:
        net = EfficientNetEncoder(mname, proj_dim = cfg.proj_dim, mlp_dim = cfg.mlp_dim, pretrained=True).to("cuda")
    elif "convnext" in mname:
        net = EfficientNetEncoder(mname, proj_dim = cfg.proj_dim, mlp_dim = cfg.mlp_dim, pretrained=False).to("cuda")
    else:
        net = ViTEncoder(mname, proj_dim=cfg.proj_dim, imsize=cfg.imsize).to("cuda")
    # if cfg.checkpoint is not None:
    #     print(f"Loading checkpoint: {cfg.checkpoint}")
    
    #     ckpt = torch.load(cfg.checkpoint,map_location="cpu")
    #     if "backbone.model.0.conv.weight" in ckpt["state_dict"].keys():
    #         print("fixing keys....")
    #         newsd = {k.replace("backbone.",""):v for k,v in ckpt["state_dict"].items() if k.startswith("backbone")}
    #         ckpt["state_dict"] = newsd

    #     net.backbone.load_state_dict(ckpt["state_dict"])
    
    #     print(
    #         f"Loaded backbone weights from model "
    #         f"{ckpt['model_name']}"
    #     )

    n_classes = len(np.unique([ds[i][1] for i in range(len(ds))]))
    probe = nn.Sequential(nn.LayerNorm(net.embed_dim), nn.Linear(net.embed_dim, n_classes)).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler() #enabled="cuda" == "cuda")
    print("model initialized")

    # Training
    meter = MetricMeter()
    for epoch in range(cfg.epochs):
        #before = net.backbone.state_dict()["model.0.conv.weight"].clone()
        net.train(), probe.train()
        for vs, y in tqdm.tqdm(train, total=len(train)):
            bs = vs.size(0)
            vs = vs.to("cuda", non_blocking=True)
            with autocast("cuda", dtype=torch.bfloat16):
                emb, proj = net(vs)
                # print("emb requires_grad:", emb.requires_grad)
                # print("proj requires_grad:", proj.requires_grad)

                # if not emb.requires_grad and not proj.requires_grad:
                #     sys.exit()
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                loss = lejepa_loss
                if do_probe:
                    y = y.to("cuda", non_blocking=True)
                    V_total = vs.shape[1]
                    y_rep = y.repeat_interleave(V_total)
                    yhat = probe(emb.detach())
                    probe_loss = F.cross_entropy(yhat, y_rep)
                    loss += probe_loss
                    pl = probe_loss.item()
                else:
                    pl = 0.0

            opt.zero_grad()
            scaler.scale(loss).backward()

            def grad_norm(model):
                total_norm = 0.0
            
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            
                return total_norm ** 0.5
            
            
            backbone_grad_norm = grad_norm(net.backbone)
            total_grad_norm = grad_norm(net)
            
            # print(f"[DEBUG] backbone grad norm: {backbone_grad_norm:.6f}")
            # print(f"[DEBUG] total grad norm:    {total_grad_norm:.6f}")
            
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            meter.update(
                sigreg_loss.item(),
                inv_loss.item(),
                lejepa_loss.item(),
                bs,
                probe = pl
            )

        # Evaluation
        #net.eval(), probe.eval()
        correct = 0
        val_sigreg = 0
        val_inv = 0
        val_probe = 0
        val_meter = MetricMeter()
        with torch.inference_mode():
            for vs, y in val:
                vs = vs.to("cuda", non_blocking=True)
                #y = y.to("cuda", non_blocking=True)
                with autocast("cuda", dtype=torch.bfloat16):
                    emb, proj = net(vs)
        
                    inv_loss = (proj.mean(0) - proj).square().mean()
                    sigreg_loss = sigreg(proj)
        
                    lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                
                    if do_probe:
                        y = y.to("cuda", non_blocking=True)
                        V_total = vs.shape[1]
                        y_rep = y.repeat_interleave(V_total)
                        yhat = probe(emb.detach())
                        probe_loss = F.cross_entropy(yhat, y_rep)
                        loss += probe_loss
                        pl = probe_loss.item()
                    else:
                        pl = 0.0
        
                val_meter.update(
                    sigreg_loss.item(),
                    inv_loss.item(),
                    #probe_loss.item(),
                    lejepa_loss.item(),
                    vs.size(0),
                    probe = pl
                )
                
            #with torch.inference_mode():
            if cfg.dataset != "GZ2":
                for vs, y in test:
                    #print("vs",vs.shape)
                    #print("y",y)
                    vs = vs.to("cuda", non_blocking=True)
                    y = y.to("cuda", non_blocking=True)
                    with autocast("cuda", dtype=torch.bfloat16):
                        emb, proj = net(vs)
                        correct += (probe(emb).argmax(1) == y).sum().item()
                        #correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
                wandb.log({"test/acc": correct / len(test_ds), "test/epoch": epoch})
        train_metrics = meter.compute()
        val_metrics = val_meter.compute()
        
        wandb.log({
            "train/sigreg": train_metrics["sigreg"],
            "train/inv": train_metrics["inv"],
            #"train/probe": train_metrics["probe"],
            "train/lejepa": train_metrics["lejepa"],
            "val/lejepa": val_metrics["lejepa"],
            "val/sigreg": val_metrics["sigreg"],
            "val/inv": val_metrics["inv"],
            #"val/probe": val_metrics["probe"],
            "epoch": epoch
        })
        if do_probe:
            wandb.log({"train/probe": train_metrics["probe"],"val/probe": val_metrics["probe"]})

        #after = net.backbone.state_dict()["model.0.conv.weight"]
        #print(f"DEBUG: weight diff: {(before - after).abs().max()}")
        torch.save({
            "model_name": cfg.model,
            "state_dict": net.state_dict(),
        }, f"{cfg.save_dir}/checkpoint.pt")
    wandb.finish()

    # os.makedirs(cfg.save_dir, exist_ok=True)

    # torch.save({
    #     "model_name": cfg.model,
    #     "state_dict": net.backbone.state_dict(),
    # }, f"{cfg.save_dir}/checkpoint_{epoch}.pt")
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()