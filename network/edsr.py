# modified from: https://github.com/thstkdgus35/EDSR-PyTorch
from argparse import Namespace
import torch.nn as nn
from NLSA import NonLocalSparseAttention

def conv_3d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def conv_2d(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
        
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class EDSR_3d(nn.Module):
    def __init__(self, args):
        super(EDSR_3d, self).__init__()
        
        # define head module
        m_head = [args.conv(args.in_channel, args.n_feats, args.kernel_size)]
        # define body module
        m_body = [ResBlock(args.conv, args.n_feats, args.kernel_size, act=args.act, res_scale=args.res_scale)
                  for _ in range(args.n_resblocks)]
        m_body.append(args.conv(args.n_feats, args.n_feats, args.kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        return res

class EDSR_atten(nn.Module):
    def __init__(self, args):
        super(EDSR_atten, self).__init__()
       
       
        # define head module
        m_head = [args.conv(args.in_channel, args.n_feats, args.kernel_size)]
        # define body module
        m_body = [NonLocalSparseAttention()]
        for _ in range(args.n_resblocks):
            m_body.append(ResBlock(args.conv, args.n_feats, args.kernel_size, act=args.act, res_scale=args.res_scale))
            
        m_body.append(NonLocalSparseAttention())
        # define tail module
        m_tail=[args.conv(args.n_feats, args.n_feats, args.kernel_size)]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.tail(self.body(x))
        res += x
        return res

def make_edsr_baseline(conv=conv_3d,n_resblocks=12, n_feats=64, res_scale=1,add_atten=False):
    args = Namespace()
    args.conv=conv
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    
    args.kernel_size = 3
    args.act = nn.ReLU(True)
    args.in_channel = 1
    print(conv)
    if add_atten:
        #print("add_atten")
        return EDSR_atten(args)
    else:
        #print("no atten")
        return EDSR_3d(args)


