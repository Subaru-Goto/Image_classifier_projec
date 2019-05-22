# import modules
import argparse
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch
from collections import OrderedDict
import json

def arguments():
    '''
    This function set parameters for a model and return them.
    '''
    parser = argparse.ArgumentParser(description = 'flower image clasifier')

    parser.add_argument('--data_directory', default ='flowers', type = str, help = 'select a file path for the training data')
    parser.add_argument('--save_dir', type = str, help = 'select a location to save a model')
    parser.add_argument('--arch', type = str, default = 'vgg', choices = ['vgg', 'alexnet'], help = 'select vgg or alexnet')
    parser.add_argument('--hidden_units', type = int, default = 2601, help = 'select a number of hidden units for the trained model')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'select a learning rate')
    parser.add_argument('--epoch', type = int, default = 6, help = 'choose epoch size')
    parser.add_argument('--gpu', type = str, default = 'cuda', choices = ['cuda', 'cpu'], help = 'select if you want to use GPU mode')

    return parser.parse_args()

# transfrom images
# https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
def data_loader(data, typ):
    '''
    This function transforms input images
    and return data sets
    '''
    # transform image data
    transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'valid': transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                'test': transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}
    data_set = {'train': datasets.ImageFolder(data + '/train', transform = transform['train']),
                 'valid': datasets.ImageFolder(data + '/valid', transform = transform['valid']),
                 'test': datasets.ImageFolder(data + '/test', transform = transform['test'])}
 
    data_loaders = {'train': torch.utils.data.DataLoader(data_set['train'], batch_size = 64, shuffle = True),
                  'valid': torch.utils.data.DataLoader(data_set['valid'], batch_size = 32, shuffle = False),
                  'test': torch.utils.data.DataLoader(data_set['test'], batch_size = 32, shuffle = False)} 
    
    return data_loaders[typ]

def train_dataset():
    '''
    This function returns a training data set.
    '''
    args = arguments()
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                    
    data_set = datasets.ImageFolder(args.data_directory + '/train', transform = transform)
    
    return data_set

def pre_model(arch, hidden):
    '''
    This function calls pre-trained model vgg or alexnet
    and adjust the last layer for the flower classification case.
    '''
    
    if arch == 'vgg':
        model = models.vgg16 (pretrained = True)
    elif arch == 'alexnet':
        model = models.alexnet (pretrained = True)
    else:
        print('please enter vgg or alexnet')
        
    # freeze pretrained model's parameters
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = {'vgg': nn.Sequential(OrderedDict([('fc', nn.Linear(25088, hidden)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p = 0.3)),
                                        ('output', nn.Linear(hidden, 102)),
                                       ('logsoftmax', nn.LogSoftmax(dim = 1))])),
                  'alexnet': nn.Sequential(OrderedDict([('fc', nn.Linear(9216, hidden)),
                                        ('relu', nn.ReLU()),
                                        ('dropout', nn.Dropout(p = 0.2)),
                                        ('output', nn.Linear(hidden, 102)),
                                       ('logsoftmax', nn.LogSoftmax(dim = 1))]))}
    # replace the last layer
    if arch == 'vgg':
        model.classifier = classifier['vgg']
        return model
    elif arch == 'alexnet':
        model.classifier = classifier['alexnet']
        return model  
        
def gpu(gpu):
    '''
    This function enables user to switch gpu and cpu
    '''
    # switch to PU or CPU
    if gpu == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    return device

def training(model, device, lrate, epochs, train, valid, arch, save_dir):
    '''
    This function trains model and print out each loss and accuracy.
    Moreover, it saves the trained model in the defined directory.
    '''
              
    # set a loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lrate)
  
    
    # set the model to gpu or cpu
    model.to(device)
    
    # initialize step
    steps = 0
    print_every = 4
    
    for epoch in range(epochs):
        # initialize running loss
        running_loss = 0
   
        for images, labels in train:
            # count steps
            steps += 1    
            # set both images and lables to GPU
            images, labels = images.to(device), labels.to(device)
        
            # initialize gradient -- reset accumulation
            optimizer.zero_grad()
        
            # forwarding
            log_p = model(images)
            loss = criterion(log_p, labels)
            loss.backward()
            optimizer.step()
        
            # sum running loss
            running_loss += loss.item()
    
    
            # initialize valid loss and accuracy
            valid_loss = 0
            accuracy = 0
        
            # print each time print every hits the set number
            if steps % print_every == 0:
                # set model to valid mode--> deactivate dropout
                model.eval()
            
                for images, labels in valid:
                    # set both images and lables to GPU
                    images, labels = images.to(device), labels.to(device)
                
                    # calc loss
                    log_p = model(images)
                    valid_loss += criterion(log_p, labels)

                    # calc accuarcy
                    # take out log
                    ps = torch.exp(log_p)
                    # get the best probability
                    top_ps, top_class = ps.topk(1, dim = 1)

                    quality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(quality.type(torch.FloatTensor))
            
                print(f"Epoch: {epoch + 1}/{epochs}..",
                     f"Training loss: {(running_loss / print_every):.3f}..",
                      f"Valid loss: {valid_loss / len(valid):.3f}..",
                      f"Valid accuracy:{accuracy / len(valid):.3f}..")
                
                running_loss = 0
                # activate dropout again
                model.train()
                
    # save the labels in the training data sets
    model.class_to_idx = train_dataset().class_to_idx
      
    # create check-points
    checkpoint = {'epochs':epochs,
                  'arch':arch,
                 'classifier':model.classifier,
                 'optimizer':optimizer.state_dict,
                  'class_to_idx':model.class_to_idx,
                 'state_dict':model.state_dict()}
    
    # save checkpoint
    if save_dir:
        torch.save(checkpoint, save_dir + '/checkpoint.pth')
    else:
        torch.save(checkpoint, 'checkpoint.pth')
        
    
    print('Done...the model is trained.')
        
