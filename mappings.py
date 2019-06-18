# Mapping functions / input nets must follow certain conventions
import copy
import torch
import torch.nn as nn
from models import Reshape
from torch.nn.modules.utils import _pair
import math

def cnn2fc(CNN, return_mask=False):
    # Given CNN returns a sequentially defined FC network with pooling/padding layers...
    # if return_mask is on, return a FC model where the zero blocks have ones 
    net_sq = list(CNN.named_children())[0][1]

    sizes = [CNN.input_size] + CNN.sizes
    size_pairs = [(sizes[i - 1][-1], sizes[i][-1]) for i in range(1, len(sizes))]

    layers = []
    masks = []
    for size, module in zip(size_pairs, net_sq):
        
        if module.__class__ == nn.Conv2d:
            assert module.padding == (0, 0)
            d_in, d_out = size
            k, s = module.kernel_size[0], module.stride[0]
            ch_in, ch_out = module.in_channels, module.out_channels
            lin_in, lin_out = ch_in*d_in*d_in, ch_out*d_out*d_out

            if return_mask:
                lin_W = torch.ones(lin_out, lin_in)
            else:
                lin_W = torch.zeros(lin_out, lin_in)

            for idx in range(ch_out):
                for i in range(d_out):
                    for j in range(d_out):
                        reverse_map = lin_W.view(ch_out, d_out, d_out, ch_in, d_in, d_in)[idx][i, j, :, :]
                        if return_mask:
                            reverse_map[:, i*s:i*s+k, j*s:j*s+k]=torch.zeros_like(module.weight[idx])
                        else:
                            reverse_map[:, i*s:i*s+k, j*s:j*s+k].copy_(module.weight[idx])
            
            with_bias = module.bias is not None
            if with_bias:
                if return_mask:
                    lin_B = torch.zeros(d_out * d_out * ch_out)
                else:
                    lin_B = module.bias.expand(d_out * d_out, ch_out).t().contiguous().view(lin_out)
                
            lin = nn.Linear(lin_in, lin_out, bias=with_bias)
            lin.weight.data.copy_(lin_W)

            if with_bias:
                lin.bias.data.copy_(lin_B)

            # append layers
            layers.append(Reshape([lin_in]))
            layers.append(lin)
            layers.append(Reshape([ch_out, d_out, d_out]))

        elif module.__class__ in [nn.ReLU,nn.Dropout,Reshape,nn.MaxPool2d,nn.ZeroPad2d]:
            layers.append(copy.deepcopy(module)) 

        elif module.__class__ == nn.Linear:
            if return_mask:
                new_module = copy.deepcopy(module)
                new_module.weight.data.zero_()
                new_module.bias.data.zero_()
                layers.append(new_module)
            else:
                layers.append(copy.deepcopy(module)) 
    return nn.Sequential(*layers)
    
def cnn2lc(CNN):
    # Given CNN returns a sequentially defined LC network with pooling/padding layers...
    net_sq = list(CNN.named_children())[0][1]

    sizes = [CNN.input_size] + CNN.sizes
    size_pairs = [(sizes[i - 1][-1], sizes[i][-1]) for i in range(1, len(sizes))]

    layers = []
    for size, module in zip(size_pairs, net_sq):
        
        if module.__class__ == nn.Conv2d:

            with_bias = module.bias is not None

            assert module.padding == (0, 0)
            d_in, d_out = size
            k, s = module.kernel_size[0], module.stride[0]
            ch_in, ch_out = module.in_channels, module.out_channels

            lc_W = torch.zeros(1, ch_out, ch_in, d_out, d_out, k**2)
            
            for idx in range(ch_out):
                for i in range(d_out):
                    for j in range(d_out):
                        lc_W[0, idx, :, i, j, :].copy_(module.weight[idx].view(ch_in, k**2))
                                    
            if with_bias:
                lc_B = module.bias.expand(d_out * d_out, ch_out).t().contiguous().view(1, ch_out, d_out, d_out)
                
            lc = LocallyConnected2d(ch_in, ch_out, d_out, k, s, with_bias)
            lc.weight.data.copy_(lc_W.view_as(lc.weight.data))

            if with_bias:
                lc.bias.data.copy_(lc_B.view_as(lc.bias.data))

            # append layers
            # layers.append(Reshape([lin_in]))
            layers.append(lc)
            # layers.append(Reshape([ch_out, d_out, d_out]))

        elif module.__class__ in [nn.ReLU,nn.Dropout,Reshape,nn.MaxPool2d,nn.ZeroPad2d,nn.Linear]:
            layers.append(copy.deepcopy(module)) 

    return nn.Sequential(*layers)

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        x = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            x += self.bias
        return x


