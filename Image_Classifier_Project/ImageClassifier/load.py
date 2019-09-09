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
from torchvision import datasets, transforms, models


def set_checkpoint_path(path):
    file_name = 'checkpoint.pth'
    complete_path = os.path.join(path,file_name)
    return complete_path

def set_path_dir(inputDirectory):
    if os.path.exists(inputDirectory):
        return inputDirectory
    else:
        print('Doesnt exists')
    
def training_model(pretrained_model, criterion, optimizer, epochs,gpu = 'gpu'):
# The training starts from here

#     epochs = 9
    print_every = 50
    steps = 0
    start = time.time()
    for e in range(epochs):
        running_loss = 0
        pretrained_model.train()

        for images, labels in trainloaders:
            steps += 1
            if gpu == 'gpu':
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                images, labels = images.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
            output = pretrained_model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                pretrained_model.eval()

                with torch.no_grad():
                    validation_loss = 0
                    accuracy = 0

                    for images, labels in validloaders:
                        if gpu == 'gpu':
                            images, labels = images.to('cuda'), labels.to('cuda')
                        else:
                            images, labels = images.to('cpu'), labels.to('cpu')
                        
                        output = pretrained_model.forward(images)
                        test_loss = criterion(output, labels)

                        validation_loss += test_loss.item()

                        ps = torch.exp(output)

                        top_p, top_class = ps.topk(1, dim=1)

                        equals = top_class = labels.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                pretrained_model.train()
                # time_elapsed = time.time() - start
#                 print("----------------TESTING LOSS CALCULATION-------------", time_elapsed)
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Training Loss: {:.3f}..".format(running_loss/print_every),
                      "Validation Loss: {:3f}..".format(validation_loss/len(validloaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloaders)))
                running_loss = 0

def validation_test(pretrained_model, testloaders):
    epochs = []
    epochs = 14
    steps = 0
    for e in range(epochs):
        test_loss = 0
        accuracy = 0

        with torch.no_grad():
            pretrained_model.eval()
            for images, labels in testloaders:
                images, labels = images.to('cuda'), labels.to('cuda')
                logps = pretrained_model(images)
                test_loss += criterion(logps, labels)

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        pretrained_model.train()

        print("Epoch: {}/{}.. ".format(e+1, epochs),
               "Test Loss: {:.3f}.. ".format(test_loss/len(testloaders)),
               "Test Accuracy: {:.3f}".format(accuracy/len(testloaders)))

# vali = validation_test(pretrained_model, testloaders)
# print("----------------VALIDATION LOSS--------------")
# print(vali)
# SAVING THE CHECKPOINT
def save_model(pretrained_model, complete_path):
    pretrained_model.class_to_idx = train_image_datasets.class_to_idx

    checkpoint = {'structure': architec,
                    'classifier':pretrained_model.classifier,
                    'state_dict': pretrained_model.state_dict(),
                    'class_to_idx': pretrained_model.class_to_idx}

    torch.save(checkpoint,complete_path)

def model(model):
    de_model = model
    return de_model
def load_model(path):
    try:
        checkpoint = torch.load(path, map_location = 'cpu')
        arch = checkpoint['structure']
        if arch.startswith("densenet121"):
            pretrained_model = models.densenet121(pretrained = True)
            input_size = pretrained_model.classifier.in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("vgg16"):
            pretrained_model = models.vgg16(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("vgg13"):
            pretrained_model = models.vgg13(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("vgg11"):
            pretrained_model = models.vgg11(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("resnet50"):
            pretrained_model = models.resnet50(pretrained = True)
            input_size = pretrained_model.fc.in_features
#         return pretrained_model, output_size, input_size
    except:
        print(path,'Incorrect PATH or No Checkpoint Available')
        print('Enter Correct Path')

        
#     pretrained_model = models.vgg16(pretrained = True)
#     checkpoint = torch.load(path)
#     pretrained_model = models.vgg16(pretrained = True)

    for i in pretrained_model.parameters():
        i.requires_grad = False
    
    pretrained_model.class_to_idx = checkpoint['class_to_idx']
    pretrained_model.classifier = checkpoint['classifier']
    pretrained_model.load_state_dict(checkpoint['state_to_dict'])

    return pretrained_model

# In case you do not want to take gpu into consideration and wanna use cpu instead
# for like loading from a CHECKPOINT

def load_model_without_cpu(path):
    try:
        checkpoint = torch.load(path, map_location = 'cpu')
        arch = checkpoint['structure']
        if arch.startswith("densenet121"):
            pretrained_model = models.densenet121(pretrained = True)
            input_size = pretrained_model.classifier.in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("vgg16"):
            pretrained_model = models.vgg16(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("vgg13"):
            pretrained_model = models.vgg13(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("vgg11"):
            pretrained_model = models.vgg11(pretrained = True)
            input_size = pretrained_model.classifier[0].in_features
#         return pretrained_model, output_size, input_size
        elif arch.startswith("resnet50"):
            pretrained_model = models.resnet50(pretrained = True)
            input_size = pretrained_model.fc.in_features
#         return pretrained_model, output_size, input_size
    except:
        print(path,'Incorrect PATH or No Checkpoint Available')
        exit()

    pretrained_model = models.vgg16(pretrained = True)

    for i in pretrained_model.parameters():
        i.requires_grad = False

    pretrained_model.class_to_idx = checkpoint['class_to_idx']
    pretrained_model.classifier = checkpoint['classifier']
    pretrained_model.load_state_dict(checkpoint['state_dict'])

    return pretrained_model

train_image_datasets, test_image_data_sets, valid_image_datasets, trainloaders, testloaders, validloaders = img.image_load()

def arg_inputdir(directory):
    inputDirectory = set_path_dir(directory)
    path_name = os.path.basename(os.path.normpath(inputDirectory))
    train_image_datasets, test_image_data_sets, valid_image_datasets, trainloaders, testloaders, validloaders = img.image_load(direc = path_name)
    return train_image_datasets, test_image_data_sets, valid_image_datasets, trainloaders, testloaders, validloaders

# name = model()
