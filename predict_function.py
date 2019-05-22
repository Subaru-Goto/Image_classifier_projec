import argparse
import json
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def pred_args():
    '''
    This function set parameters for a model.
    '''
    parser = argparse.ArgumentParser(description = 'parameter setting for predicting a flower image')

    parser.add_argument('image_path', type = str, help = 'select a file path to an image data')
    parser.add_argument('checkpoint', type = str, help = 'please choose a saved trained model')
    parser.add_argument('--top_k', type = int, default = 5, help = 'please enter integer number for how many ranking you would like to see')
    parser.add_argument('--category_name', default = 'cat_to_name.json', type = str, help = 'please define a json file for a catagory name')
    parser.add_argument('--gpu', type = str, default = 'cuda', choices = ['cuda', 'cpu'], help = 'select if you want to use GPU mode')

    return parser.parse_args()


def read_json():
    '''
    This function reads a json file and load the labels into dictionary
    '''
    pred_arg = pred_args()
    
    with open(pred_arg.category_name, 'r') as f:
        return json.load(f)

def load_checkpoint(save_path, gpu):
    '''
    This function receives a path of saved model and gpu or cpu mode.
    It loads the model.
    '''
    # load the saved model' info
    checkpoint = torch.load(save_path)
    # pre_trained model
    if checkpoint['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else:
        model = models.vgg16 (pretrained = True)
        
    # freeze the pre_trained model's parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # load info from saved checkpoint
    epochs = checkpoint['epochs']
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.state_dict = checkpoint['state_dict']
    
    # change to GPU
    device = torch.device('cuda' if gpu == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)
        
    return model 

def process_image(image):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model.
    Returns an Numpy array
    '''
    # set the shortest size
    px_size = 225
    # open an image data
    test_image = Image.open(image)
    # get the size
    width, height = test_image.size
    # calc with ratio of the image
    with_ratio = width / height
    # if with ratio is bigger than 1 , it means the image is wider.
    if with_ratio > 1:
        test_image = test_image.resize((round(px_size * with_ratio), px_size))
    else: # height is bigger, so divide the px with the ratio
        test_image = test_image.resize((px_size, round(px_size / with_ratio)))
  
    # crop out the center 224x224
    # https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    
    width, height = test_image.size
    new_width = 224
    new_height = 224
    
    left = round((width - new_width)/2)
    top = round((height - new_height)/2)
    right = round((width + new_width)/2)
    bottom = round((height + new_height)/2)

    test_image = test_image.crop((left, top, right, bottom))
    
    # convert the color channel to 0-1 range by dividing by the max value
    np_image = np.array(test_image) / 255
    # normalize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean) / std
    
    # reorder color channel // width, height, color to color, width, height
    # https://stackoverflow.com/questions/32034237/how-does-numpys-transpose-method-permute-the-axes-of-an-array
    np_image = np_image.transpose(2, 0, 1)
    
    # convert numpy to tensor
    return torch.from_numpy(np_image)

def predict(image_path, model, gpu, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        
        # set the model to GPU or CPU    
        device = torch.device('cuda' if gpu == 'cuda' and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        # load a image
        image = process_image(image_path)
        # set the image to GPU or CPU
        image = image.to(device)
        # reshape the image to the trained model
        reshaped_img = image.unsqueeze(0)
        reshaped_img = reshaped_img.float()
        # get probability with logsoftmax
        log_p = model(reshaped_img)
        # remove logarithm
        ps = torch.exp(log_p)
        # get the top 5 probabilities and classes
        probs, idx = ps.topk(topk, dim = 1)
        
        # change tensor to numpy
        probs = np.array(probs)[0]
        idx = np.array(idx)[0]
        
        # convert index to class
        index = []        
        for i in range(len(idx)):
            idx_list = np.array(idx)[i]
            # get key from value # https://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
            index_to_key = [cls for cls, idx in model.class_to_idx.items() if idx == idx_list][0]
            index.append(index_to_key)
            
        # convert index to category name
        # call read json function
        cat_to_name = read_json()
        name = []        
        for i in range(len(index)):
            idx = index[i]
            name.append(cat_to_name[str(idx)])

        return print(probs), print(name)
