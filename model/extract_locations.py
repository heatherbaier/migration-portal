from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, utils, models
from model import RecurrentAttention
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import AverageMeter
from config import get_config
from trainer import Trainer
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision
import random
import shutil
import pickle
import torch
import utils
import json
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_inputs(impath):
    to_tens = transforms.ToTensor()
    return to_tens(Image.open(impath).convert('RGB')).unsqueeze(0)



m = open("../../pooling/data/migration_data.json",)
mig_data = json.load(m)
m.close()
mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
mig_data.columns = ['muni_id', 'num_migrants']
q = 2
mig_data['class'] = pd.qcut(mig_data['num_migrants'], q = q, labels = [i for i in range(q)])
mig_data



def get_png_names(directory):
    images = []
    for i in os.listdir(directory):
        try:
            if os.path.isdir(os.path.join(directory, i)):
                new_path = os.path.join(directory, i, "pngs")
                image = os.listdir(new_path)[0]
                images.append(os.path.join(directory, i, "pngs", image))
        except:
            pass
    return images


            

image_names = get_png_names("../../attn/data/MEX/")


y_class, y_mig = [], []

for i in image_names:
        dta = mig_data[mig_data["muni_id"] == i.split("/")[5]]
        if len(dta) != 0:
            y_class.append(dta['class'].values[0])
            y_mig.append(dta['num_migrants'].values[0])



train_num = int(25 * .70)

train_indices = random.sample(range(0, 25), train_num)
val_indices = [i for i in range(0, 25) if i not in train_indices]




batch_size = 1
train = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y_class[i], y_mig[i]) for i in train_indices]
val = [(torchvision.transforms.functional.adjust_brightness(load_inputs(image_names[i]), brightness_factor = 2).squeeze(), y_class[i], y_mig[i]) for i in val_indices]
train_dl = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)
            
            
config, unparsed = get_config()                    
trainer = Trainer(config, (train_dl, val_dl))    


checkpoint = torch.load("./ckpt/ram_4_50x50_0.75_model_best.pth.tar")
checkpoint = checkpoint["model_state"]
            
            



import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import denormalize, bounding_box


def denormalize(T, coords):
    return 0.5 * ((coords + 1.0) * T)


def exceeds(from_x, to_x, from_y, to_y, H, W):
    """Check whether the extracted patch will exceed
    the boundaries of the image of size `T`.
    """
    if (from_x < 0) or (from_y < 0) or (to_x > H) or (to_y > W):
        return True
    return False


def fix(from_x, to_x, from_y, to_y, H, W, size):

    """
    Check whether the extracted patch will exceed
    the boundaries of the image of size `T`.
    If it will exceed, make a list of the offending reasons and fix them
    """

    offenders = []

    if (from_x < 0):
        offenders.append("negative x")
    if from_y < 0:
        offenders.append("negative y")
    if from_x > H:
        offenders.append("from_x exceeds h")            
    if to_x > H:
        offenders.append("to_x exceeds h")
    if from_y > W:
        offenders.append("from_y exceeds w")
    if to_y > W:
        offenders.append("to_y exceeds w")            


    if ("from_y exceeds w" in offenders) or ("to_y exceeds w" in offenders):
        from_y, to_y = W - size, W

    if ("from_x exceeds h" in offenders) or ("to_x exceeds h" in offenders):
        from_x, to_x = H - size, H     

    elif ("negative x" in offenders):
        from_x, to_x = 0, 0 + size

    elif ("negative y" in offenders):
        from_y, to_y = 0, 0 + size            

    return from_x, to_x, from_y, to_y




locations_dict = {}

for i in image_names:
    
    print(i)
    
    muni_id = i.split("/")[5]
    image = load_inputs(i)
    locations = trainer.extract_locations(image, checkpoint)
    
    start = [denormalize(image.shape[2], l) for l in locations]
    start = torch.cat([start[l].unsqueeze(0) for l in range(4)])
        
    B, C, H, W = image.shape
    
    size = int(min(H, W) / 5)
    
    end = start + size
    
#     print(start, end)
    
    coords_dict = {}
    
    for c in range(0, len(start)):
        
        from_coords = start[c]
        to_coords = end[c]
        
        from_x = from_coords[0][0].item()
        from_y = from_coords[0][1].item()
        
        to_x = to_coords[0][0].item()
        to_y = to_coords[0][1] .item()   
        
        if exceeds(from_x = from_x, to_x = to_x, from_y = from_y, to_y = to_y, H = H, W = W):
        
            from_x, to_x, from_y, to_y = fix(from_x = from_x, to_x = to_x, from_y = from_y, to_y = to_y, H = H, W = W, size = size)
                
        coords_dict['glimpse' + str(c)] = {'from_x': from_x, 'to_x': to_x, 'from_y': from_y, 'to_y': to_y}
    
    locations_dict[i] = coords_dict
    
    
    
with open("locations_extract.json", "w") as outfile: 
    json.dump(locations_dict, outfile)