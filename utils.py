import torch
import torch.nn as nn
import numpy as np
import os
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import time
import shutil
import random

def normalize(img):
    if img.max()-img.min()==0:
        print(img.min(),img.max())
    return (img - img.min()) / (img.max() - img.min())

def crop_bg(image):
    contour = 4
    
    minVal = image[0,0,0]
    threshold = minVal+0.001
    foreground = image > threshold
    (x,) = np.nonzero(np.amax(foreground, axis=(1,2)))
    (y,) = np.nonzero(np.amax(foreground, axis=(0,2)))
    (z,) = np.nonzero(np.amax(foreground, axis=(0,1)))
    
    x_min=x.min() - contour if x.min() > contour else 0
    y_min=y.min() - contour if y.min() > contour else 0
    z_min=z.min() - contour if z.min() > contour else 0
    
    x_max=x.max()+contour if x.max()+contour<image.shape[0] else image.shape[0]
    y_max=y.max()+contour if y.max()+contour<image.shape[1] else image.shape[1]
    z_max=z.max()+contour if z.max()+contour<image.shape[2] else image.shape[2]
    
    #print(y_min,y_max)            
    return image[x_min:x_max,y_min:y_max,z_min:z_max]
    
def padding(data,shape):
    pad = [(0, 0)] * data.ndim
    for i in range(data.ndim):
        if shape[i] > data.shape[i]:
            w_before = (shape[i] - data.shape[i]) // 2
            w_after = shape[i] - data.shape[i] - w_before
            
            pad[i] = (w_before, w_after)
    data = np.pad(data, pad_width=pad, mode='constant', constant_values=0) 
    return data
    
def center_crop(data, shape, mask):
    data=padding(data,shape)
    w,h,d=data.shape
    assert w>=shape[0] and h>=shape[1] and d>=shape[2]
    w_start=(w-shape[0])//2
    h_start=(h-shape[1])//2
    d_start=(d-shape[2])//2
    crop=data[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]
    crop_mask=mask[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]     
    return {'crop':crop,'crop_mask':crop_mask}
        

def random_crop(data,shape, mask):
    w,h,d=data.shape
    #print(data.shape,shape)
    assert w>=shape[0] and h>=shape[1] and d>=shape[2]
    while 1:
        w_start=random.randrange(w-shape[0])
        h_start=random.randrange(h-shape[1])
        if (d-shape[2])==0:
            d_start=0
        else:
            d_start=random.randrange(d-shape[2])
        crop=data[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]
        crop_mask=mask[w_start:w_start+shape[0],h_start:h_start+shape[1],d_start:d_start+shape[2]]  
        if crop.max()>crop.min():
            break
    return {'crop':crop,'crop_mask':crop_mask}
    
def make_coord(shape, ranges=None, flatten=False):
    """ Make coordinates at grid centers.
    """
    
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (n-1)
        seq = v0 +  r * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  #  H x W x D x 3
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def make_coord_2(inp_shape,shape, spacing_ratio, ranges=None, flatten=False):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(inp_shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
       
        r = (v1 - v0) / ((n-1)*spacing_ratio[i])
        seq = v0 + r * torch.arange(shape[i]).float()
        #print(seq)
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)  #  H x W x D x 3
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
     
def input_matrix_wpn(outH,outW,outD, scale,flatten=False):
    
    # projection_pixel_coordinate (H,W,2) coordinate(i,j)=[[i/r],[j/r]]
    h_p_coord = torch.arange(0, outH, 1).float().mul(1.0 / scale[0])
    h_p_coord_ = torch.floor(h_p_coord).int().view(outH,1,1)
    h_p_coord_metrix = h_p_coord_.expand(outH, outW,outD).unsqueeze(3)

    w_p_coord = torch.arange(0, outW, 1).float().mul(1.0 / scale[1])
    w_p_coord_ = torch.floor(w_p_coord).int().view(1,outW,1)
    w_p_coord_metrix = w_p_coord_.expand(outH, outW,outD).unsqueeze(3)

    d_p_coord = torch.arange(0, outD, 1).float().mul(1.0 / scale[2])
    d_p_coord_ = torch.floor(d_p_coord).int().view(1, 1, outD)
    d_p_coord_metrix = d_p_coord_.expand(outH, outW, outD).unsqueeze(3)

    projection_coord= torch.cat([h_p_coord_metrix, w_p_coord_metrix,d_p_coord_metrix], dim=-1)
    
    if flatten:
        projection_coord=projection_coord.view(-1,3)    
    return projection_coord    # HxWxD,3     
    
def to_pixel_samples(img,scale):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    #coord = make_coord(img.shape)   # shape=(HxWxDx3)
    #H,W,D=img.shape
    #center_coord=coord[H//4:H//4*3,W//4:W//4*3,D//4:D//4*3,:].reshape(-1,3)
    #center_value = img[H//4:H//4*3,W//4:W//4*3,D//4:D//4*3].reshape(-1,1)   # shape=(H*W*D,1)
    #return center_coord, center_value
    coord = make_coord(img.shape,flatten=True)   # shape=(H*W*D,3)
    value = img.reshape(-1,1)   # shape=(H*W*D,1)
    proj_coord=input_matrix_wpn(*img.shape, scale,flatten=True)
    return coord, value,proj_coord
    
def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
    
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size,conv=default_conv, stride=1, bias=True,
        bn=False, act=nn.PReLU()):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
            
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_save_path(save_path):
    ensure_path(save_path)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer



def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer
