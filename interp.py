import argparse
import copy
import gc
import models
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from main import evaluate
from torchvision import datasets, transforms
from mappings import cnn2fc
from utils import get_id, get_data, accuracy
from utils import load_net
import os

def get_interpolated_net(alpha, net1, net2, interp_method='linear'):

    interp_net = copy.deepcopy(net1)
    
    zipped_params = zip(interp_net.parameters(), net1.parameters(), net2.parameters())

    if interp_method == 'linear':
        for section_no, (p_interp, p_init, p_final) in enumerate(zipped_params):
            p_interp.data.copy_((1.0 - alpha) * p_init.data + alpha * p_final.data)
        return interp_net

    elif interp_method == 'curvilinear':
        for section_no, (p_interp, p_init, p_final) in enumerate(zipped_params):
            vector = (1.0 - alpha) * p_init.data + alpha * p_final.data
            radius = (1.0 - alpha) * p_init.data.norm() + alpha * p_final.data.norm()
            p_interp.data.copy_(vector*radius / vector.norm())
            
    return interp_net


def train_string(train_loader, points, optimizers, stiffness, crit, device):

    for point in points[1:-1]:
        point.train()

    hist_epoch = []
    
    spring_total_loss = 0
    landscape_total_loss = 0
    total_loss = 0
    
    for x, y in train_loader:

        spring_loss = 0
        landscape_loss = 0
        loss = 0

        for i in range(1, len(points) - 1): #keep the endpoints fixed
            opt = optimizers[i - 1]

            # loop over dataset
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = points[i](x)
            
            landscape_loss_point = crit(out, y)
            landscape_loss += landscape_loss_point.item()
            landscape_loss_point.backward()
            
            opt.step()
            
            for current, before, after in zip(points[i].parameters(), points[i - 1].parameters(), points[i + 1].parameters()):
                with torch.no_grad():
                    spring_loss += 0.5 * stiffness * ((current-before).norm().pow(2) + (current-after).norm().pow(2)).detach().item()
                    current.data += stiffness * (after.data + before.data - 2 * current.data)
                #NEB
                # tangent = after.data - before.data
                # tangent /= tangent.norm()
                # current.grad -= torch.dot(current.grad.view(-1),tangent.view(-1))*tangent

                torch.cuda.empty_cache()
                gc.collect()
                
        loss = landscape_loss + spring_loss
        print('landscape loss : ', landscape_loss)
        print('spring loss : ', spring_loss)
        print('combined loss : ', loss)
        hist_epoch.append(loss)

    total_loss += loss            

    return hist_epoch

def get_interpolated_string(net1, net2, train_loader, n_points, lr, mom, wd, stiffness, device, crit, epochs):

    points = [net1]
    optimizers = []
    alphas = np.linspace(0, 1, n_points)

    #initialize
    for i in range(1, n_points-1):
        print('initializing point ',i)
        new_point = get_interpolated_net(alphas[i], net1, net2)
        new_optimizer =  optim.SGD(new_point.parameters(),
                                   lr=lr,
                                   momentum=mom,
                                   weight_decay=wd)
        points.append(new_point)
        optimizers.append(new_optimizer)
        
    points.append(net2)
        
    #train
    training_history = []
    for epoch in range(epochs):
        hist_epoch = train_string(train_loader, points, optimizers, stiffness, crit, device)
        print('epoch {} : loss {}\n'.format(epoch, np.mean(hist_epoch)))
        training_history.append(hist_epoch)
        
    return points, training_history

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net1_path', default=None, type=str, required=True)
    parser.add_argument('--net2_path', default=None, type=str, required=True)
    parser.add_argument('--interp_method', default='linear', type=str, choices=['linear', 'curvilinear', 'average', 'string'])
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--n_points', default=5, type=int)
    parser.add_argument('--stiffness', default=1, type=float)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--data_size', default=0, type=int)
    parser.add_argument('--bs_train', default=50, type=int)
    parser.add_argument('--bs_eval', default=50, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--data_path', default='~/data', type=str)
    parser.add_argument('--log_path', default='./log', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no_cuda', action='store_true', default=False)

    args = parser.parse_args()
    args.path = get_id(args.data_path)
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)
    
    # training setup
    train_loader, tr_loader_eval, te_loader_eval, num_classes = get_data(args.dataset, args.data_path, args.bs_train, args.bs_eval, args.data_size)
    crit = nn.CrossEntropyLoss().to(args.device)

    net1 = load_net(args.net1_path, num_classes, args.device)
    net2 = load_net(args.net2_path, num_classes, args.device)
    
    if args.interp_method == 'string':

        points, training_history = get_interpolated_string(net1, net2, train_loader, args.n_points, 
                                         args.lr, args.mom, args.wd, args.stiffness, 
                                         args.device, crit, args.epochs)
        print('Trained the string successfully!')
        
    alphas = np.linspace(0, 1, args.n_points)
    tr_evals = []
    te_evals = []
    
    for i,alpha in enumerate(alphas):

        print('evaluating point {} out of {}'.format(i,len(alphas)))
        if args.interp_method == 'string':
            net = points[i]
        elif args.interp_method in ['linear', 'curvilinear']:
            net = get_interpolated_net(alpha, net1, net2, args.interp_method)
        elif args.interp_method == 'average':
            AverageNet = getattr(models,'AverageNet')
            net = AverageNet(alpha, net1, net2).to(args.device) #

        tr_evals.append(evaluate(tr_loader_eval, net, crit, args.device))
        te_evals.append(evaluate(te_loader_eval, net, crit, args.device))
    
    print('train\n', tr_evals) 
    print('test\n', te_evals)

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    prefix = '{}_{}_{}'.format(args.net1_path.split('/')[-1].split('.')[0],
                               args.net2_path.split('/')[-1].split('.')[0],
                               args.interp_method)
        

    if args.interp_method == 'string':
        torch.save(training_history,args.log_path+'/{}.hist'.format(prefix))
    torch.save(tr_evals,args.log_path+'/{}.train'.format(prefix))
    torch.save(te_evals,args.log_path+'/{}.test'.format(prefix))
