#  First Pre-process the image
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import time
import json
from collections import OrderedDict
import seaborn as sns
import argparse
import load
import img


parser = argparse.ArgumentParser(description='This is predict function')
parser.add_argument('inputDirectory_image',help=' Enter directory path of the image',action='store',default = '/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg')
parser.add_argument('checkpoint_path',help='Enter the checkpoint path', action='store',default='/home/workspace/ImageClassifier/checkpoint.pth')
parser.add_argument('--category_names',dest='category',default = 'cat_to_name.json')
parser.add_argument('--top_k', dest = 'topk', type=int,default = 5)
parser.add_argument('--gpu',dest='gpu_use',action = 'store_true')

args = parser.parse_args()
checkpoint = args.checkpoint_path
inputdir = args.inputDirectory_image
category_name = args.category
topk_value = args.topk
gpu_u = args.gpu_use

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

# PREDICTION TIME !
with open(category_name) as f:
        flower_to_name = json.load(f)
        
def predict(image, pretrained_model,flower_to_name, topk_value):
    processed_image = img.process_image(image)
    pretrained_model.to('cpu')
    processed_image.unsqueeze_(0)
    probs = torch.exp(pretrained_model.forward(processed_image))
    top_probs, top_labs = probs.topk(topk_value)

    top_probs = top_probs.detach().numpy().tolist()
    top_labs = top_labs.tolist()

    labels = pd.DataFrame({'class':pd.Series(pretrained_model.class_to_idx),'flower_name':pd.Series(flower_to_name)})
    labels = labels.set_index('class')
    labels = labels.iloc[top_labs[0]]
    labels['predictions'] = top_probs[0]

    return labels

# img = (data_dir +'/test' + '/27/' + 'image_06864.jpg')
# val = predict(img, pretrained_model)
# print(val)

def sanity(img):
    labels = predict(img, pretrained_model)
    plt.figure(figsize=(5,10))
    ax = plt.subplot(2,1,1)

    img - process_image(img)
    imshow(img,ax)
    sns.set_style("whitegrid")
    sns.subplot(2,1,2)
    sns.barplot(x = labels['predictions'], y = labels['flower_name'], color = '#047495')
    plt.xlabel("Probability of Prediction")
    plt.xlabel("")
    plt.show();
sns.set_style("white")

# model = load.model()
# print(model)
if gpu_u:
    checkpoint = load.load_model(checkpoint)
    arch_name = checkpoint['structure']
    labels = predict(inputdir,pretrained_model,flower_to_name,topk_value)
#     print(labels)
else:
    pretrained_model = load.load_model_without_cpu(checkpoint)
# pretrained_model = train.pretrained_model
    labels = predict(inputdir,pretrained_model,flower_to_name,topk_value)
    print(labels)
