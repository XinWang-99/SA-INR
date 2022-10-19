import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class AttentionLayer(torch.nn.Module):
    def __init__(self, n_feats, win_size,layerType,dilation=1):
        super(AttentionLayer, self).__init__()
        self.win_size=win_size
        self.layerType=layerType
        assert layerType=='PBLA' or layerType=='FBLA'
        if layerType=='PBLA':
            self.pos_embedding = nn.Sequential(nn.Linear(3, 256),nn.Linear(256, 1))
        else:
            self.q=nn.Linear(n_feats,n_feats)
            self.k=nn.Linear(n_feats,n_feats)
        self.v=nn.Linear(n_feats,n_feats)
        
        self.dilation=dilation    
        self.pad = torch.tensor([win_size[i]//2 for i in range(3)])  # (3,3,1)
        self.pad_list =  torch.tensor([self.win_size[2]//2,self.win_size[2]//2, self.win_size[1]//2, self.win_size[1]//2,self.win_size[0]//2,self.win_size[0]//2])  # (1,1,3,3,3,3)
        self.offset_coord = self.get_offset_coord(win_size)
        self.softmax = nn.Softmax(dim=-1)
        
        
    def projection(self, matrix, coord):
        # matrix (b,64,H,W,D)  coord(b,N,3)
        b, N = coord.shape[:2]
        ind = torch.arange(0, b).view(b, 1).expand(b, N).contiguous().view(-1)
        # return (b,N,64)
        # print(coord[:,:,0].max(),coord[:,:,1].max(),coord[:,:,2].max())
        # print(matrix.shape)
        return matrix[ind, :, coord[:, :, 0].view(-1), coord[:, :, 1].view(-1), coord[:, :, 2].view(-1)].view(b, N, -1)

    def get_offset_coord(self, win_size):
        half_win_size = [math.ceil(win_size[i] * 0.5) - 1 for i in range(3)]
        offset = [torch.arange(-half_win_size[i], win_size[i] - half_win_size[i], 1) for i in range(3)]
        # print(offset[1].shape)
        offset1 = offset[1].unsqueeze(0).repeat(win_size[2], 1).permute(1, 0).reshape(-1, 1)
        offset2 = offset[2].unsqueeze(1).repeat(win_size[1], 1)
        offset12 = torch.cat([offset1, offset2], dim=1)
        offset0 = offset[0].unsqueeze(0).repeat(win_size[1] * win_size[2], 1).permute(1, 0).reshape(-1, 1)
        offset12 = offset12.repeat(win_size[0], 1)
        offset012 = torch.cat([offset0, offset12], dim=1)
        return offset012 
           
    def local_attention(self, q_feat,feat, proj_coord,hr_coord):  # feat (b,c,w/2,h/2,d/2) q_feat(b,N,64) proj_coord(b,N,3) hr_coord (b,N,3)
        win_size = self.win_size
        b, c = feat.shape[:2]
        N =proj_coord.shape[1]
        assert N>0
        pad = self.pad.clone().to(feat.device)
        pad[:2]=pad[:2]*self.dilation   # (3,3,1)-> # (3*dilation,3*dilation,1)
        pad = pad.view(1, 1, 3).expand(b, N, 3)
        proj_coord = proj_coord + pad

        offset_coord = self.offset_coord.clone().to(feat.device)
        offset_coord[:,:2]=offset_coord[:,:2]*self.dilation
        offset_coord = offset_coord.view(1, 1, *offset_coord.shape).expand(b, N, *offset_coord.shape)
        local_coord = proj_coord.unsqueeze(2).expand(b, N, win_size[0] * win_size[1] * win_size[2], 3)+\
                      offset_coord  # (b,N,win_size[0]*win_size[1]*win_size[2],3)

        local_coord = local_coord.view(b, -1, 3).type(torch.long)  # (b,N*win_size[0]*win_size[1]*win_size[2],3)

        # padding feature map

        pad_list=self.pad_list.clone()
        pad_list[2:]=pad_list[2:]*self.dilation
        padding = nn.ReplicationPad3d(tuple(pad_list.tolist()))
        
        padded_feat = padding(feat)
        local_feat = self.projection(padded_feat, local_coord).view(b, -1, win_size[0] * win_size[1] * win_size[2],
                                                                    c)  # (b,N,win_size[0]*win_size[1]*win_size[2],64)

        if self.layerType=='PBLA':   
            lr_coord = make_coord(feat.shape[2:]).permute(3, 0, 1, 2).to(feat.device)       
            lr_coord = lr_coord.unsqueeze(0).expand(b, *lr_coord.shape)  # (b,3,w/2,h/2,d/2)
            padded_coord = padding(lr_coord)
            local_pos = self.projection(padded_coord, local_coord).view(b, -1, win_size[0] * win_size[1] * win_size[2],
                                                                        3)  # (b,N,win_size[0]*win_size[1]*win_size[2],3)
            relative_pos = local_pos - hr_coord.unsqueeze(-2)  # (b,N,win_size[0]*win_size[1]*win_size[2],3)
            embedded_pos = self.pos_embedding(relative_pos)  # (b,N,win_size[0]*win_size[1]*win_size[2],1)
           
            atten=embedded_pos.squeeze(-1)
        else:
            q=self.q(q_feat)
            k=self.k(local_feat)
            atten = torch.einsum('ijk,ijmk->ijm', q,k)  # (b,N,win_size[0]*win_size[1]*win_size[2])
        atten = self.softmax(atten)
        v=self.v(local_feat)
        x = torch.einsum('ijm,ijmk->ijk', atten, v)
        return x 
        
    def forward(self, q_feat,feat, proj_coord,hr_coord):
        a = self.local_attention(q_feat,feat, proj_coord,hr_coord)
     
        return q_feat+a
