# train_yolo_full.py

import os
from ultralytics import YOLO
import sys
#sys.path.append('/home/users/l/lastufka')
#from feuerzeug.datasets.COCODataset import MGCLSodDataset
import wandb
from ultralytics.data.dataset import YOLODataset
from ultralytics import YOLO
from ultralytics.data.build import build_yolo_dataset
import cv2
import numpy as np
from ultralytics.data.loaders import LoadImagesAndLabels

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "/home/users/l/lastufka/scratch/MGCLS_data/enhanced/test_data_prep_cs"
TRAIN_JSON = f"{DATASET_PATH}/train/mgcls_coco_annotations_train.json"
VAL_JSON   = f"{DATASET_PATH}/test/mgcls_coco_annotations_test.json"

DATA_YAML = f"{DATASET_PATH}/dataset.yaml"

MODEL_NAME = "yolov8m.pt"   # <-- "medium" model (closest to what you asked)

EPOCHS = 50
IMGSZ = 640
BATCH = 16
DEVICE = 0  # set -1 for CPU

class MGCLSYOLODataset(YOLODataset):

    def load_image(self, i, rect_mode=True):

        path = self.im_files[i]
        print("CUSTOM LOAD:", path)
        # -----------------------------
        # NPY LOADING
        # -----------------------------
        if path.endswith(".npy"):

            img = np.load(path)

            img = cv2.normalize(
                img,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            # grayscale -> HWC
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)

            # 1 channel -> 3 channel
            if img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)

            img = np.ascontiguousarray(img)

            h0, w0 = img.shape[:2]

            return img, (h0, w0), img.shape[:2]

        # default behavior
        return super().load_image(i, rect_mode)

def custom_build_yolo_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    return MGCLSYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=rect,
        cache=cfg.cache,
        single_cls=cfg.single_cls,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=f"{mode}: ",
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction,
    )

def on_train_batch_start(trainer):
    batch = trainer.train_loader
    imgs = batch["img"] if isinstance(batch, dict) else batch[0]
    print("BATCH SHAPE:", imgs.shape)
# -----------------------------
# 2. LOAD MODEL
# -----------------------------
def load_model():
    model = YOLO(MODEL_NAME)
    print(f"[INFO] Loaded model: {MODEL_NAME}")
    return model


# -----------------------------
# 3. TRAIN
# -----------------------------
def train(model):
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project="mgcls_yolo",
        name="mgcls_yolo_training",
        cache=False,
    )
    return results


# -----------------------------
# 4. VALIDATE
# -----------------------------
def validate(model):
    metrics = model.val()
    print("[INFO] Validation complete")
    return metrics


# -----------------------------
# 5. INFERENCE TEST
# -----------------------------
def inference(model):
    test_image = f"{DATASET_PATH}/test/images"

    results = model.predict(
        source=test_image,
        save=True,
        conf=0.25
    )

    print("[INFO] Inference complete. Results saved in runs/")
    return results



# -----------------------------
# MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("=== YOLO FULL PIPELINE START ===")

    # Step 1: YAML
    #create_yaml()
    loader = LoadImagesAndLabels(f"{DATASET_PATH}/train_5k/images", img_size=640)

    img, label, path, shapes = loader[0]
    
    print("TYPE:", type(img))
    print("SHAPE:", img.shape)
    print("PATH:", path)
    #build.build_yolo_dataset = custom_build_yolo_dataset
    # Step 2: Model
    model = load_model()
    #model.add_callback("on_train_batch_start", on_train_batch_start)

    train(model)

    # Step 4: Validate
    validate(model)

    # Step 5: Inference
    inference(model)

    print("=== DONE ===")