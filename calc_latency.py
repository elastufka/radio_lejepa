# X = embeddings from source domain (N x D)
# Y = embeddings from target domain (M x D)

import torch
import numpy as np
import sys
#from PIL import Image
#sys.path.append("/home/users/l/lastufka")
#from feuerzeug.hf_utils import extract_features, get_dataset
#from feuerzeug.plotting import plot_dataset_sample
#from FM_compare.finetune import get_model
import argparse
import time
import timm


def load_model(model_name, checkpoint=None, img_size=224):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        img_size=img_size,
    )

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)

    model.eval().cuda()

    return model


def benchmark(
    model,
    image_size=224,
    warmup=100,
    iters=1000,
    dtype=torch.float16,
):

    x = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device="cuda",
        dtype=dtype,
    )

    # warmup
    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=dtype):
                _ = model(x)

    torch.cuda.synchronize()

    times = []

    with torch.inference_mode():

        for _ in range(iters):

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.autocast("cuda", dtype=dtype):
                _ = model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append(end - start)

    times = np.asarray(times)

    print(f"Model: {type(model).__name__}")
    print(f"Image size: {image_size}")
    print(f"Iterations: {iters}")
    print()

    print(f"Mean latency   : {times.mean()*1000:.3f} ms")
    print(f"Median latency : {np.median(times)*1000:.3f} ms")
    print(f"Std            : {times.std()*1000:.3f} ms")
    print(f"FPS            : {1/times.mean():.2f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    
    print(args.model)
    model = load_model(
        args.model,
        checkpoint=args.checkpoint,
        img_size=args.img_size,
    )

    benchmark(model,image_size=args.img_size,warmup=args.warmup,iters=args.iters,)