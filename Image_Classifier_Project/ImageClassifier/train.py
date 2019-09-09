# Importing all the libraries that are required
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
# import time
from collections import OrderedDict
import argparse
import img
import model
import sys
import os
import time
import torch
from torch import nn, optim
import os.path
import load

# Creating Parser
parser = argparse.ArgumentParser(description = "Command line argument parser")
# parser.add_argument('--save_dir',dest='save_dir',action='store', help="Location to save the model checkpoint')
parser.add_argument('inputDirectory',help='Path to the input directory.',action='store', default = None)
parser.add_argument('--save_dir', dest = "save_dir", action = "store",default = '/home/workspace/ImageClassifier')
parser.add_argument('-a','--arch', dest = 'arch', default = "vgg16", type = str)
parser.add_argument('--gpu',dest='gpu',default = 'gpu',action = 'store_true')
hyper_param = parser.add_argument_group('hyperparameters')
hyper_param.add_argument('-l', '--learning_rate',dest = 'l',help="Specify your LEarning rate value",action='store', default = 0.001, type = float)
hyper_param.add_argument('-u','--hidden_units', default = 3136,dest = 'u',type = int)
hyper_param.add_argument('-e','--epochs', dest = 'e',default = 8, type = int)

args = parser.parse_args()

data_d = args.inputDirectory
architec = args.arch
lr = args.l
hidden_units = args.u
epochs = args.e
gpu = args.gpu
path = args.save_dir


train_image_datasets, test_image_data_sets, valid_image_datasets, trainloaders, testloaders, validloaders = load.arg_inputdir(args.inputDirectory)
pretrained_model, output_size, i = model.arch(architec)
pretrained_model, criterion, optimizer= model.classifier(pretrained_model, output_size,i,lr, hidden_units)
load.training_model(pretrained_model,criterion, optimizer,epochs ,gpu)
load.validation_test(pretrained_model, testloaders)
complete_path  = load.set_checkpoint_path(path)
load.save_model(pretrained_model,complete_path,architec)
load.model(architec)
# pretrained_model = load.load_model(pretrained_model,'checkpoint.pth')