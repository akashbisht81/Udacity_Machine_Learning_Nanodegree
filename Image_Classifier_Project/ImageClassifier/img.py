from torchvision import datasets, transforms, models
import torch
from PIL import Image
# data_dir = "flowers"
# train_dir = data_dir +"/train"
# valid_dir = data_dir +"/valid"
# test_dir =  data_dir +"/test"

def image_load(direc = None):
    if direc is not None:
        data_dir = direc
        train_dir = data_dir +"/train"
        valid_dir = data_dir +"/valid"
        test_dir =  data_dir +"/test"
    elif direc is None:
        data_dir = "flowers"
        train_dir = data_dir +"/train"
        valid_dir = data_dir +"/valid"
        test_dir =  data_dir +"/test"
        
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                          [0.485,0.456,0.406],                                                         [0.229,0.224, 0.225])
                                         ])
    validation_data_transforms  = transforms.Compose([transforms.RandomResizedCrop(224),
                                                      transforms.ToTensor(),
                                                   transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) ])



    testing_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                                                      ])

    train_image_datasets = datasets.ImageFolder(root = train_dir, transform = data_transforms)
    test_image_datasets = datasets.ImageFolder(root = test_dir, transform = testing_data_transforms)
    valid_image_datasets = datasets.ImageFolder(root = valid_dir, transform = validation_data_transforms)

    trainloaders = torch.utils.data.DataLoader(train_image_datasets,batch_size = 32, shuffle = True)
    testloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = 32)
    validloaders = torch.utils.data.DataLoader(valid_image_datasets, shuffle = True)
    
    return train_image_datasets, test_image_datasets, valid_image_datasets, trainloaders, testloaders, validloaders


def process_image(image):
    pil_image = Image.open(image)

    image_trans = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                      transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229,0.224,0.225])
                                      ])
    image_fin = image_trans(pil_image)
    return image_fin
# img = (data_dir +'/test'+'/27/'+'image_06864.jpg')
# img = process_image(img)
