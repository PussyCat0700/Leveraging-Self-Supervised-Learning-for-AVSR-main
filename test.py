import os
import shutil
import sys  # 导入sys模块
import fairseq
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.plugins import *
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_warmup_lr import WarmupLR

from config import args
from data.lrs2_dataset import LRS2
from data.utils import collate_fn
from models.moco_visual_frontend import MoCoVisualFrontend
from models.utils import PositionalEncoding, conv1dLayers, outputConv, MaskedLayerNorm, generate_square_subsequent_mask
from utils.decoders import ctc_greedy_decode, teacher_forcing_attention_decode
from utils.label_smoothing import SmoothCTCLoss, SmoothCrossEntropyLoss
from utils.metrics import compute_error_ch, compute_error_word
from scheduler import  WarmupReduceLROnPlateau

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
#optimizer = optim.Adam(self.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))

# import h5py
#
# f= h5py.File( "/home/gryang/LRS3/LRS3.h5", "r")
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].shape)
#     for i in range(10):
#         print(f[key][i].shape )
#         print(f[key][i] )

# with open("/home/xcpan/LRS2/mvlrs_v1/pretrain.txt","r") as f:
#     data = f.readlines()
# print("pretrain",len(data))
#
# with open("/home/xcpan/LRS2/mvlrs_v1/val.txt","r") as f:
#     data = f.readlines()
# print("val",len(data))
#
# with open("/home/xcpan/LRS2/mvlrs_v1/test.txt","r") as f:
#     data = f.readlines()
# print("test",len(data))
#
#
# with open("/home/xcpan/LRS2/mvlrs_v1/train.txt","r") as f:
#     data = f.readlines()
# print("train",len(data))


# with open("/home/gryang/LRS3/pretrain.txt","r") as f:
#     data = f.readlines()
# print("pretrain",len(data))
#
# with open("/home/gryang/LRS3/val.txt","r") as f:
#     data = f.readlines()
# print("val",len(data))
#
# with open("/home/gryang/LRS3/test.txt","r") as f:
#     data = f.readlines()
# print("test",len(data))
#
#
# with open("/home/gryang/LRS3/train.txt","r") as f:
#     data = f.readlines()
# print("train",len(data))
#
# exit(0)
#
# a="THE WORST FINANCIAL CRISIS IN A GENERATION WHY THREE WELL THREE IS THE MAGIC NUMBER IN RHETORIC GOVERNMENT OF THE PEOPLE BY THE PEOPLE FOR THE PEOPLE {NS} {NS}"
# # a="THE FRUIT OFFERED BY EVE AND INSTEAD EATEN {a} THE SNAKE PROTEIN {LG} THE WORLD WOULD BE DIFFERENT {LG}"
# # a="abc"
# coun=a.count("{")
# for i in range(coun):
#     left=a.find("{")
#     if left!=-1:
#         right=a.find("}")
#         a= a.replace(a[left:right+2],"")
# a=a.strip()
# a = a.split(" ")
# print(a)

