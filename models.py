import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import copy

class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape # must be a list

    def __repr__(self):
        return ('Reshape({})'.format(self.shape))    
        
    def forward(self, x):
        self.bs = x.size(0)
        return x.view(self.bs, *self.shape)

######################### SKINNYNET #########################

class SkinnyCNN(nn.Module):
    '''
    A rather small convnet
    '''
    def __init__(self, input_size=[3, 32, 32], ch=64, num_classes=10):
        super(SkinnyCNN, self).__init__()
        
        self.input_channels = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]

        self.features = nn.Sequential(

            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        
        self.size = self.get_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.size, num_classes, bias=True),
        )

    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.features(x)
        return y.view(-1).size(0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SkinnyCNNPrime(nn.Module):

    def __init__(self, input_size=[3, 32, 32], ch=64, num_classes=10):
        super(SkinnyCNNPrime, self).__init__()
        
        self.input_size = torch.Size(input_size)
        self.ch = ch

        layers = []

        layers.append(nn.ZeroPad2d(padding=2))
        layers.append(nn.Conv2d(self.input_size[0], self.ch, kernel_size=4, stride=2)) 
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.ZeroPad2d(padding=2))
        layers.append(nn.Conv2d(self.ch, self.ch, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        layers.append(nn.ZeroPad2d(padding=1))
        layers.append(nn.Conv2d(self.ch, self.ch, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.size = self.get_size(layers)
        layers.append(Reshape([self.size]))
        layers.append(nn.Linear(self.size, num_classes, bias=True))
        self.net = nn.Sequential(*layers)

        self.sizes = self.get_sizes() 


    def get_size(self, layers):
        # hack to get the size for the FC transition layer...
        x = torch.randn(1, *self.input_size)
        net = nn.Sequential(*layers)
        y = net(x)
        return y.view(-1).size(0)
    
    def get_sizes(self):
        # hack to get all sizes
        res = []
        han = [l.register_forward_hook(lambda m, i, o: res.append(o.size())) for l in self.net]
        self.net(torch.randn(1, *self.input_size))
        [h.remove() for h in han]
        return res 

    def forward(self, x):
        return self.net(x)


def SkinnyCNNPrime_FC(num_classes=10): #note that this is a function, not a class
    layers = []
    layers.append(nn.ZeroPad2d(padding=2))
    layers.append(Reshape([3888]))
    layers.append(nn.Linear(3888, 18496, bias=True))
    layers.append(Reshape([64,17,17]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    
    layers.append(nn.ZeroPad2d(padding=2))
    layers.append(Reshape([9216]))
    layers.append(nn.Linear(9216, 4096, bias=True))
    layers.append(Reshape([64,8,8]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    
    layers.append(nn.ZeroPad2d(padding=1))
    layers.append(Reshape([1600]))
    layers.append(nn.Linear(1600, 576, bias=True))
    layers.append(Reshape([64,3,3]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    
    layers.append(Reshape([64]))
    layers.append(nn.Linear(64, num_classes, bias=True))

    return nn.Sequential(*layers)


######################### ALEXNET #########################

class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256)
        x = self.classifier(x)
        return x


class AlexNetPrime(nn.Module):
    ''' 
    Identical copy of AlexNet above with the required layer conventions
    This convention follows the order in what model.named_children returns
    All paddings within CONV layers must be ZERO, they are separate layers now
    Input size can be a list or torch.Size both result in torch.Size
    ''' 
    def __init__(self, input_size=[3, 32, 32], num_classes=10):
        super(AlexNetPrime, self).__init__()

        layers = []
        layers.append(nn.ZeroPad2d(padding=2))
        layers.append(nn.Conv2d(3, 64, kernel_size=4, stride=2)) 
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(nn.ZeroPad2d(padding=2))
        layers.append(nn.Conv2d(64, 192, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(nn.ZeroPad2d(padding=1))
        layers.append(nn.Conv2d(192, 384, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ZeroPad2d(padding=1))
        layers.append(nn.Conv2d(384, 256, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ZeroPad2d(padding=1))
        layers.append(nn.Conv2d(256, 256, kernel_size=3))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(Reshape([256])) # New module!
        layers.append(nn.Dropout())
        layers.append(nn.Linear(256, 512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout())
        layers.append(nn.Linear(512, 512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(512, num_classes))

        self.net = nn.Sequential(*layers)

        self.input_size = torch.Size(input_size)
        self.sizes = self.get_sizes()

    def forward(self, x):
        return self.net(x)

    def get_sizes(self):
        res = []
        han = [l.register_forward_hook(lambda m, i, o: res.append(o.size())) for l in self.net]
        self.net(torch.randn(1, *self.input_size))
        [h.remove() for h in han]
        return res 


def AlexNetPrime_FC(num_classes=10): #note that this is a function, not a class
    layers = []
    layers.append(nn.ZeroPad2d(padding=2))
    layers.append(Reshape([3888]))
    layers.append(nn.Linear(3888, 18496, bias=True))
    layers.append(Reshape([64,17,17]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    
    layers.append(nn.ZeroPad2d(padding=2))
    layers.append(Reshape([9216]))
    layers.append(nn.Linear(9216, 12288, bias=True))
    layers.append(Reshape([192,8,8]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    
    layers.append(nn.ZeroPad2d(padding=1))
    layers.append(Reshape([4800]))
    layers.append(nn.Linear(4800, 3456, bias=True))
    layers.append(Reshape([384,3,3]))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.ZeroPad2d(padding=1))
    layers.append(Reshape([9600]))
    layers.append(nn.Linear(9600, 2304, bias=True))
    layers.append(Reshape([256,3,3]))
    layers.append(nn.ReLU(inplace=True))

    layers.append(nn.ZeroPad2d(padding=1))
    layers.append(Reshape([6400]))
    layers.append(nn.Linear(6400, 2304, bias=True))
    layers.append(Reshape([256,3,3]))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
    
    layers.append(Reshape([256]))
    layers.append(nn.Dropout())
    layers.append(nn.Linear(256, 512, bias=True))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Dropout(p=0.5))
    layers.append(nn.Linear(512, 512, bias=True))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(512, num_classes, bias=True))

    return nn.Sequential(*layers)

######################### EXPERIMENT #########################

class SkinnyCNNplusFC(nn.Module):

    def __init__(self, input_size=[3, 32, 32], ch=64, num_classes=10):
        # ch is the scale factor for number of channels
        super(SkinnyCNNplusFC, self).__init__()
        
        self.input_channels = input_size[0]
        self.input_height = input_size[1]
        self.input_width = input_size[2]

        self.cnn_features = nn.Sequential(

            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(ch, ch, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        
        self.size = self.get_size()
        self.cnn_classifier = nn.Sequential(
            nn.Linear(self.size, num_classes, bias=True),
        )

        fc_layers = []
        fc_layers.append(nn.ZeroPad2d(padding=2))
        fc_layers.append(Reshape([3888]))
        fc_layers.append(nn.Linear(3888, 18496, bias=True))
        fc_layers.append(Reshape([64,17,17]))
        fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        fc_layers.append(nn.ZeroPad2d(padding=2))
        fc_layers.append(Reshape([9216]))
        fc_layers.append(nn.Linear(9216, 4096, bias=True))
        fc_layers.append(Reshape([64,8,8]))
        fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        fc_layers.append(nn.ZeroPad2d(padding=1))
        fc_layers.append(Reshape([1600]))
        fc_layers.append(nn.Linear(1600, 576, bias=True))
        fc_layers.append(Reshape([64,3,3]))
        fc_layers.append(nn.ReLU(inplace=True))
        fc_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        
        fc_layers.append(Reshape([64]))
        fc_layers.append(nn.Linear(64, num_classes, bias=True))

        self.fc_part = nn.Sequential(*fc_layers)

        
    def get_size(self):
        # hack to get the size for the FC layer...
        x = torch.randn(1, self.input_channels, self.input_height, self.input_width)
        y = self.cnn_features(x)
        return y.view(-1).size(0)

    def forward(self, x):
        cnn_out = self.cnn_features(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        cnn_out = self.cnn_classifier(cnn_out)
        fc_out  = self.fc_part(x)
        return cnn_out + fc_out
    

######################### FOR ENSEMBLING #########################

class AverageNet(nn.Module):
    ''' 
    Identical copy of AlexNet above with the required layer conventions
    This convention follows the order in what model.named_children returns
    All paddings within CONV layers must be ZERO, they are separate layers now
    Input size can be a list or torch.Size both result in torch.Size
    ''' 
    def __init__(self, alpha, net1, net2):
        super(AverageNet, self).__init__()

        self.alpha = alpha
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        return (1.-self.alpha) * self.net1(x) + self.alpha * self.net2(x)



def alexnetprime(**kwargs):
    return AlexNetPrime(**kwargs)

def alexnetprime_fc_version(**kwargs):
    return AlexNetPrime_FC(**kwargs)

def skinnyprime(**kwargs):
    return SkinnyCNNPrime(**kwargs)

def skinnyprime_fc_version(**kwargs):
    return SkinnyCNNPrime_FC(**kwargs)

def skinnycnnplusfc(**kwargs):
    return SkinnyCNNplusFC(**kwargs)

    
