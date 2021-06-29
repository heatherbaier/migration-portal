from utils import AverageMeter, load_inputs, get_png_names, get_ys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, utils, models
from model import RecurrentAttention
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import get_config
from utils import plot_images
from trainer import Trainer
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torchvision
import random
import pickle
import shutil
import torch
import utils
import json
import time

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


m = open("../../pooling/data/migration_data.json",)
mig_data = json.load(m)
m.close()
mig_data = pd.DataFrame.from_dict(mig_data, orient = 'index').reset_index()
mig_data.columns = ['muni_id', 'num_migrants']
q = 4
mig_data['class'] = pd.qcut(mig_data['num_migrants'], q = q, labels = [i for i in range(q)])
mig_data


image_names = get_png_names("../../attn/data/MEX/")
y_class, y_mig = get_ys(image_names, mig_data)


train_num = int(25 * .70)
train_indices = random.sample(range(0, 25), train_num)
val_indices = [i for i in range(0, 25) if i not in train_indices]


class miniConv(torch.nn.Module):
    
    def __init__(self, resnet):
        super().__init__()
        
        h_g = 128     
        D_in = 1 * 128 * 1 * 1
        self.conv1_miniConv = resnet.conv1.to(device)
        self.bn1_miniConv = resnet.bn1.to(device)
        self.relu_miniConv = resnet.relu.to(device)
        self.maxpool_miniConv = resnet.maxpool.to(device)
        self.layer1_miniConv = resnet.layer1.to(device)
        self.layer2_miniConv = resnet.layer2.to(device)
        self.adp_pool_miniConv = torch.nn.AdaptiveAvgPool2d((1, 1)).to(device)        
        self.fc1 = torch.nn.Linear(D_in, h_g)#.to('cuda:1')
        
    def forward(self, phi):
        
        phi = self.conv1_miniConv(phi)
        phi = self.bn1_miniConv(phi)
        phi = self.relu_miniConv(phi)
        phi = self.maxpool_miniConv(phi)
        phi = self.layer1_miniConv(phi)
        phi = self.layer2_miniConv(phi)
        phi = self.adp_pool_miniConv(phi)
        phi = phi.flatten(start_dim = 1)  # Keep a batch_size = num_glimpses for predictions at each scale
        phi_out = F.relu(self.fc1(phi)) # feed phi to respective fc layer
        return phi
    
    
device = "cuda"
resnet = torchvision.models.resnet18()
miniConv_model = miniConv(resnet = resnet).to(device)

checkpoint = torch.load("./ckpt/ram_4_50x50_0.75_model_best.pth.tar")
checkpoint = checkpoint["model_state"]

matching_keys = [p for p in list(checkpoint.keys()) if "_miniConv" in p]
matching_keys2 = [p for p in list(checkpoint.keys()) if p == 'sensor.fc1.weight' or p == 'sensor.fc1.bias']
[matching_keys2.append(k) for k in matching_keys]

weights = {}
for mk in matching_keys2:
    miniConv_model_key = mk.replace("sensor.", "")
    weights[miniConv_model_key] = checkpoint[mk]
    
miniConv_model.load_state_dict(weights)


g = open("./locations_extract.json")
glimpses = json.load(g)
g.close()



features_dict = {}

for im_name, patches in glimpses.items():
        
    muni_id = im_name.split("/")[5]
    print(muni_id)
    image = load_inputs(im_name)
        
    cur_features_dict = {}
    for glimpseID, coords in patches.items():
                
        from_x, to_x, from_y, to_y = int(coords["from_x"]), int(coords["to_x"]), int(coords["from_y"]), int(coords["to_y"])
        cur_patch = image[:, :, from_x:to_x, from_y:to_y]

        features = miniConv_model(cur_patch.to(device))[0].detach().cpu().numpy()
        features = [str(f) for f in features]
        cur_features_dict[glimpseID] = features

    features_dict[im_name] = cur_features_dict
                

with open("features_extract.json", "w") as outfile: 
    json.dump(features_dict, outfile)