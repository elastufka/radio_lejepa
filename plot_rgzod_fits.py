import sys
import os
import numpy as np
sys.path.append("/home/users/l/lastufka")
from feuerzeug.plotting import *
from feuerzeug.utils import fits_loader
from feuerzeug.datasets.FITSDataset import *
from astropy.io import fits
from PIL import Image
import glob

if __name__ == "__main__":
    #ds = FITSFolderDataset("/home/users/l/lastufka/scratch/rgz/od/FITSImages/FITSImages")
    #fig=plot_dataset_sample(ds)
    #fig.savefig("/home/users/l/lastufka/eupe_paper/rgzod_fits_sample.png")
    os.chdir("/home/users/l/lastufka/scratch/rgz/od/test/images")
    aa = glob.glob("*.png")
    os.chdir("/home/users/l/lastufka/scratch/rgz/od/FITSImages/FITSImages")
    for d in aa:
        fname = d[:-3]
        with fits.open(f"{d[:-3]}fits") as f:
            d=f[0].data
        d = np.nan_to_num(d, nan=0.0)
        d = d - np.min(d)
        if d.max() > 0:
            d = d / np.max(d)
        print(np.min(d),np.max(d))
        img = Image.fromarray((255 * d).astype(np.uint8), mode="L").transpose(Image.FLIP_TOP_BOTTOM)
        img.save(f"/home/users/l/lastufka/scratch/rgz/od/PNGImagesFromFITS/{fname}png")