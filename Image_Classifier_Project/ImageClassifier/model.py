from torchvision import  models
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

import json
import img
 
def arch(arch = "vgg16"):
    with open('cat_to_name.json') as f:
        flower_to_name = json.load(f)
    output_size = len(flower_to_name)
    try:
        if arch.startswith("densenet121"):
            pretrained_model = models.densenet121(pretrained = True)
            input_size = pretrained_model.classifier.in_features
            return pretrained_model, output_size, input_size
        if arch.startswith("vgg16"):
            pretrained_model = models.vgg16(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
            return pretrained_model, output_size, input_size
        if arch.startswith("vgg13"):
            pretrained_model = models.vgg13(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
            return pretrained_model, output_size, input_size
        if arch.startswith("vgg11"):
            pretrained_model = models.vgg11(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
            return pretrained_model, output_size, input_size
        if arch.startswith("resnet50"):
            pretrained_model = models.resnet50(pretrained = True)
            input_size = pretrained_model.fc.in_features
            return pretrained_model, output_size, input_size
    except:
        print('Please Choose from the following:--')
        print('[resnet50,vgg11,vgg13,vgg16,densenet121]')

def classifier(pretrained_model,output_size,input_size,lr,hidden_units):
    
    new_classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(hidden_units,784),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.00),
                                    nn.Linear(784, output_size),
                                    nn.LogSoftmax(dim=1))
    pretrained_model.classifier = new_classifier

    pretrained_model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(pretrained_model.classifier.parameters(), lr = lr)
    return pretrained_model, criterion, optimizer