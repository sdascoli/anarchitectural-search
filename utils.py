# some useful functions
import os
import models
import torch
from torchvision import datasets, transforms
from importlib import import_module
import torch.optim as optim
from mappings import cnn2fc

def get_id(path):
    import subprocess
    whoami = subprocess.run(['whoami'], stdout=subprocess.PIPE)
    whoami = whoami.stdout.decode('utf-8')
    whoami = whoami.strip('\n')
    if whoami == 'lds351':
        path = '/beegfs/lds351'
    elif whoami == 'leventsagun':
        path = '~/data'
    return path

def load_net(net_path, num_classes, device):
    net_state = torch.load(net_path, map_location=device)
    saved_args_path = os.path.dirname(net_path)
    saved_args = torch.load(saved_args_path+'/args.pyT')
    net_model = saved_args.model
    model_class = getattr(models, net_model)
    print(net_path, model_class)
    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(net_state['weights'])
    del net_state
    if 'fc' not in net_model:
        model = cnn2fc(model).to(device)
    return model

def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    # here one must add .item() otherwise GPU calculation gives roundoff errors!!!
    # and total acc at the end might appear at more than 100 with small prec error
    return 100 * correct.sum().float().item() / y.size(0)
    # alternatives...
    # return 100 * correct.sum().double().item() / y.size(0)
    # return 100 * correct.sum().item() / y.size(0)

def get_data(dataset, path, bs_train, bs_eval, data_size=0):
    # mean/std stats
    if dataset == 'cifar10':
        num_classes = 10
        stats = {
            'mean': [0.491, 0.482, 0.447], 
            'std': [0.247, 0.243, 0.262]
            } 
    elif dataset == 'cifar100':
        num_classes = 100
        stats = {
            'mean': [0.49137, 0.48235, 0.44667], 
            'std': [0.24706, 0.24353, 0.26157]
            }    
    elif dataset == 'mnist':
        num_classes = 10
        stats = {
            'mean': [0.1307], 
            'std': [0.3081]
            } 


    # input transformation w/o preprocessing for now
    trans = [
        transforms.ToTensor(),
        lambda t: t.type(torch.get_default_dtype()),
        transforms.Normalize(**stats)
        ]

    # get tr and te data with the same normalization
    # no preprocessing for now
    tr_data = datasets.__dict__[dataset.upper()](
        root=path, 
        train=True, 
        download=True,
        transform=transforms.Compose(trans)
        )

    te_data = datasets.__dict__[dataset.upper()](
        root=path, 
        train=False, 
        download=True,
        transform=transforms.Compose(trans)
        )
    
    if data_size > 0: #get smaller dataset
        ds_train = torch.utils.data.DataLoader(dataset=tr_data,batch_size=data_size, shuffle=False)
        ds_test = torch.utils.data.DataLoader(dataset=te_data,batch_size=data_size, shuffle=False)
        for small_x, small_y in ds_train:
            break
        tr_data_small = list(zip(small_x, small_y))
        ds_test = torch.utils.data.DataLoader(dataset=te_data,batch_size=data_size, shuffle=False)
        for small_x, small_y in ds_test:
            break
        te_data_small = list(zip(small_x, small_y))
        train_loader = torch.utils.data.DataLoader(dataset=tr_data_small,batch_size=bs_train, shuffle=False)
        train_loader_eval = torch.utils.data.DataLoader(dataset=tr_data_small,batch_size=bs_eval, shuffle=False)
        test_loader_eval = torch.utils.data.DataLoader(dataset=te_data_small,batch_size=bs_eval, shuffle=False)
        
    else:
        train_loader = torch.utils.data.DataLoader(dataset=tr_data, batch_size=bs_train, shuffle=False)
        train_loader_eval = torch.utils.data.DataLoader(dataset=tr_data, batch_size=bs_eval, shuffle=False)
        test_loader_eval = torch.utils.data.DataLoader(dataset=te_data, batch_size=bs_eval,shuffle=False)
                          
    return train_loader, train_loader_eval, test_loader_eval, num_classes

