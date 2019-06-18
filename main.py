import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models
from torchvision import datasets, transforms
from mappings import cnn2fc, cnn2lc
from utils import get_id, get_data, accuracy
import pickle
import sys
import gc

def train(train_loader, net, crit, opt, device, print_freq=40):

    net.train()

    hist_loss = []
    hist_acc = []
    
    for batch_idx, (x, y) in enumerate(train_loader):
        # loop over dataset
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        # record history (starts at initial point)
        hist_loss.append(loss.item())
        hist_acc.append(accuracy(out, y))
        
        if batch_idx % print_freq == 0:
            print('Iteration {}/{} \t Loss : {:10.3f} \t Accuracy : {:10.3f}'.format(batch_idx, len(list(train_loader)), hist_loss[-1], hist_acc[-1]))

        # take a step
        loss.backward()
        opt.step()

        gc.collect()
    
    return hist_loss, hist_acc


def evaluate(eval_loader, net, crit, device):

    net.eval()

    with torch.no_grad(): # alt. just call backward to free memory
        total_size = 0
        total_loss = 0
        total_acc = 0
        for x, y in eval_loader:
            # loop over dataset
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = crit(out, y).item()
            prec = accuracy(out, y)
            bs = x.size(0)
            
            total_size += bs
            total_loss += loss * bs
            total_acc += prec * bs
            
    return [total_loss / total_size, total_acc / total_size] 


if __name__ == '__main__':
    
    t_init = time.time()

    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--path', default='~/data', type=str,
                        help='path to data')
    parser.add_argument('--data_size', default=0, type=int,
                        help='set to nonzero value to reduce size of dataset')
    # MODEL
    parser.add_argument('--model', default='alexnetprime', type=str, choices=['alexnetprime', 'skinnyprime'])
    parser.add_argument('--convert_to', default='nil', type=str, choices=['nil','fc','lc'])
    parser.add_argument('--load_model', default='', type=str,
                        help='path to the model to be loaded')
    parser.add_argument('--save_dir', default='results', type=str,
                        help='path to save run')

    # TRAINING
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--n_saves', default=20, type=int)
    parser.add_argument('--bs_train', default=250, type=int)
    parser.add_argument('--bs_eval', default=1000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)

    # OTHER
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--load_opt', action='store_true', default=False)

    args = parser.parse_args()

    print(args)

    # initial setup
    args.path = get_id(args.path)
    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    torch.manual_seed(args.seed)

    train_loader, tr_loader_eval, te_loader_eval, num_classes = get_data(args.dataset, args.path, args.bs_train, args.bs_eval, args.data_size)

    # get/load the model, optimizer, and crit
    if args.load_model != '':
        state = torch.load(args.load_model) # gives the state_dict and opt
        args.model = args.load_model.split("/")[-1].split("_")[0] # this is by our saving convention
        model_class = getattr(models, args.model)
        net = model_class(num_classes=num_classes).to(args.device)
        net.load_state_dict(state['weights'])
    else:
        model_class = getattr(models, args.model)
        net = model_class(num_classes=num_classes).to(args.device)
    
    if args.convert_to=='fc':
        net = cnn2fc(net).to(args.device)
        args.model += '_fc_version' 

    if args.convert_to=='lc':
        net = cnn2lc(net).to(args.device)
        args.model += '_lc_version' 

    p = pickle.dumps(net)
    size = sys.getsizeof(p)
    print('size of model in bytes : ', size)
        
    opt = optim.SGD(
        net.parameters(),
        lr=args.lr, 
        momentum=args.mom,
        weight_decay=args.wd
        )

    if args.load_opt: 
        # load the optimizer state from file
        # must take place after net and opt are defined in order
        opt.load_state_dict(state['optimizer'])

    crit = nn.CrossEntropyLoss().to(args.device)

    print(net)

    checkpoints = np.unique(np.logspace(0, np.log10(args.epochs), args.n_saves).astype(int))
    checkpoints = np.insert(checkpoints, 0, 0) # add the initial point
    print('checkpoints: {}'.format(checkpoints))
    
    # save the initial state and args, assumes folder exists
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    state = {'weights': net.state_dict(), 'optimizer': opt.state_dict()}
    torch.save(state, args.save_dir + '/{}_0.pyT'.format(args.model))
    torch.save(args, args.save_dir + '/args.pyT')

    # training process
    training_history = {'tr_step_loss': [], 'tr_step_acc': []}
    evaluation_history = {'train_loss': [], 'test_loss': [],
                          'train_acc': [], 'test_acc': []
                          }
    training_history['checkpoints'] = checkpoints
    evaluation_history['checkpoints'] = checkpoints
    
    time_mem_history = {'train': [], 'eval': [], 'total': '', 'mem': []}

    # initial performance
    te_epoch = evaluate(te_loader_eval, net, crit, args.device)
    print('Test loss and accuracy :', te_epoch)
    tr_epoch = evaluate(tr_loader_eval, net, crit, args.device)
    print('Train loss and accuracy :', tr_epoch)
    evaluation_history['train_loss'].append(tr_epoch[0])
    evaluation_history['train_acc'].append(tr_epoch[1])
    evaluation_history['test_loss'].append(te_epoch[0])
    evaluation_history['test_acc'].append(te_epoch[1])
    state = {'weights': net.state_dict(), 'optimizer': opt.state_dict()}
    torch.save(state, args.save_dir + '/{}_0.pyT'.format(args.model))

    for epoch in range(args.epochs):
    
        print('epoch {} begins'.format(epoch + 1))

        t = time.time()
        step_loss, step_acc = train(train_loader, net, crit, opt, args.device)
        training_history['tr_step_loss'] += step_loss
        training_history['tr_step_acc'] += step_acc
        time_mem_history['train'].append((epoch + 1, '{:3f}'.format(time.time() - t)))

        if (epoch + 1) in checkpoints:
            t = time.time()
            te_epoch = evaluate(te_loader_eval, net, crit, args.device)
            print('Test loss and accuracy :', te_epoch)
            tr_epoch = evaluate(tr_loader_eval, net, crit, args.device)
            print('Train loss and accuracy :', tr_epoch)
            evaluation_history['train_loss'].append(tr_epoch[0])
            evaluation_history['train_acc'].append(tr_epoch[1])
            evaluation_history['test_loss'].append(te_epoch[0])
            evaluation_history['test_acc'].append(te_epoch[1])
                
            time_mem_history['eval'].append((epoch + 1, '{:3f}'.format(time.time() - t)))

            if args.device == 'cuda':
                time_mem_history['mem'].append((torch.cuda.memory_allocated() / (1024**2), 
                                            torch.cuda.memory_cached() / (1024**2)))
            
            state = {'weights': net.state_dict(), 'optimizer': opt.state_dict()}
            torch.save(state, args.save_dir + '/{}_{}.pyT'.format(args.model, epoch + 1))
            torch.save(training_history, args.save_dir + '/training_history.hist')
            torch.save(evaluation_history, args.save_dir + '/evaluation_history.hist')
            time_mem_history['total'] = '{:3f}'.format(time.time() - t_init)
            torch.save(time_mem_history, args.save_dir + '/time_mem_history.hist')

