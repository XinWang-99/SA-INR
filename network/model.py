import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from edsr import conv_2d, conv_3d, make_edsr_baseline
from mlp import MLP
from NLSA import NonLocalSparseAttention
from utils import make_coord,input_matrix_wpn,to_pixel_samples
from AttentionLayer import AttentionLayer

class NLSALayer(nn.Module):
    def __init__(self, n_feats):
        super(NLSALayer, self).__init__()
        
        self.atten=NonLocalSparseAttention(channels=n_feats)  
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(n_feats,n_feats,  kernel_size=3,padding=1, bias=True)
             
    def forward(self, x):
        x=self.atten(x)
        a = self.conv(self.relu(x))
        return x+a
        
class FFNLayer(nn.Module):
    def __init__(self, n_feats):
        super(FFNLayer, self).__init__()
        
        self.fc1 = nn.Linear(n_feats,n_feats)
        self.fc2 = nn.Linear(n_feats,n_feats)
        
        self.norm = nn.LayerNorm(n_feats)
        
    def forward(self, x):

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm(x + a)
        
        return x
        
class SA_INR(nn.Module):
    def __init__(self, conv=conv_3d, n_resblocks=8, n_feats=64, win_size=(7, 7, 2),layerType='FBLA', dilation=1\
                     ,add_res=False,add_NLSA=False,add_branch=False,is_train=True):
        super().__init__()
        self.add_res=add_res
        self.add_NLSA=add_NLSA
        self.add_branch=add_branch
        self.is_train=is_train
        self.encoder = make_edsr_baseline(conv, n_resblocks, n_feats, add_atten=False)
        
        if self.add_NLSA:    
            self.NLSAlayer=NLSALayer(n_feats)       
        if self.add_branch:
            self.mask_predictor=nn.Linear(n_feats,2)
        self.attentionLayer=AttentionLayer(n_feats,win_size,layerType,dilation)
        self.imnet = MLP(in_dim=n_feats, out_dim=1, hidden_list=[256, 256, 256, 256])
    
    def get_feat(self,inp):
        
        feat =self.encoder(inp)  # feat (b,c,w/2,h/2,d/2)
        if self.add_NLSA:  
            feat=self.NLSAlayer(feat)
        return feat
        
    def inference(self, inp, feat,hr_coord,proj_coord):  # inp (b,1,w/2,h/2,d/2)  # hr_coord (b,w*h*d,3) # proj_coord (b,w*h*d,3)
        
        
        n_feats=feat.shape[1]

        bs, sample_q = hr_coord.shape[:2]
        
        # feat (bs,c,w,h,d)   hr_coord (bs,sample_q,3)   sample_q:??????
        q_feat = F.grid_sample(feat, hr_coord.flip(-1).view(bs, 1, 1, sample_q, 3), mode='bilinear',
                               align_corners=True)[:, :, 0, 0, :].permute(0, 2, 1)  # q_feat (bs,c,sample_q)->(bs,sample_q,c)
        
        if not self.add_branch:
            mask=torch.cat([torch.zeros((bs, sample_q,1)),torch.ones((bs, sample_q,1))],dim=-1).to(inp.device)
        else:  
            mask_p=self.mask_predictor(q_feat)  # mask (b,n,2)
            if self.is_train:              
                mask=F.gumbel_softmax(mask_p,tau=1,hard=True,dim=2)  # return 2-dim one-hot tensor
            else:
                mask=F.softmax(mask_p,dim=2)
                mask=(mask>0.5).float()

        #print(torch.sum(mask[:,:,0]),torch.sum(mask[:,:,1]))
        # branch 1
        idx_easy= torch.nonzero(mask[:,:,0].view(-1)).squeeze(1)   # idx_easy (m1)
        if torch.sum(mask[:,:,0])>0:
            feat_easy=torch.index_select(q_feat.contiguous().view(-1,q_feat.shape[-1]),0,idx_easy)   # feat_easy (m1,c)
            #feat_easy=self.FFNLayer1(feat_easy)
            pred_easy=self.imnet(feat_easy) # pred_easy (m1,1)
         
        # branch 2  
        idx_difficult= torch.nonzero(mask[:,:,1].view(-1)).squeeze(1)   # idx_difficult (m2)
        if torch.sum(mask[:,:,1])>0:
            pred_difficult=[]
            for i in range(bs): 
                if torch.sum(mask[i,:,1])==0: 
                    continue          
                idx_each= torch.nonzero(mask[i,:,1].view(-1)).squeeze(1)  # idx_each (m2/)      
                hr_coord_each=torch.index_select(hr_coord[i],0,idx_each).unsqueeze(0)  # hr_coord_each(1,m2/,3)
                proj_coord_each=torch.index_select(proj_coord[i],0,idx_each).unsqueeze(0)  # proj_coord_each (1,m2/,3)
                
                q_feat_each=torch.index_select(q_feat[i],0,idx_each).unsqueeze(0)  # feat_each (1,m2/,c)
                feat_each=self.attentionLayer(q_feat_each,feat[i].unsqueeze(0), proj_coord_each, hr_coord_each).squeeze(0)
                #feat_each=self.FFNLayer2(feat_each)
                pred_each=self.imnet(feat_each)  # (m2/,1)
                pred_difficult.append(pred_each)
            
            pred_difficult=torch.cat(pred_difficult,dim=0)  # pred_difficult (m2,1)
        # combine and scatter
        pred=torch.zeros(bs*sample_q, 1).cuda()
        if torch.sum(mask[:,:,0])==0:
            idx=idx_difficult.unsqueeze(1)
            pred_shuffled=pred_difficult
        elif torch.sum(mask[:,:,1])==0:
            idx=idx_easy.unsqueeze(1)
            pred_shuffled=pred_easy
        else:     
            idx=torch.cat([idx_easy,idx_difficult]).unsqueeze(1)   # idx (n,1)
            pred_shuffled=torch.cat([pred_easy,pred_difficult])
    
        pred.scatter_(0, idx,pred_shuffled)
        pred=pred.unsqueeze(0).view(bs,sample_q,1)
        #print(pred.shape)
        if self.add_res:
            ip = F.grid_sample(inp, hr_coord.flip(-1).view(bs, 1, 1, sample_q, 3), mode='bilinear', align_corners=True)[
                 :, :, 0, 0, :].permute(0, 2, 1)  # ip (b,w*h*d,1)
            
            pred += ip
        
        #sparsity=(0.68*torch.sum(mask[:,:,0])+torch.sum(mask[:,:,1]))/(bs*sample_q)
        sparsity=torch.sum(mask[:,:,0])/(bs*sample_q)

            
        return {"pred":pred,
                "sparsity":sparsity,
                "mask":mask[:,:,1]}
      
    def forward(self, inp,hr_coord,proj_coord):  # inp (b,1,w/2,h/2,d/2)  # hr_coord (b,w*h*d,3) # proj_coord (b,w*h*d,3)
        feat=self.get_feat(inp)
        return self.inference(inp,feat,hr_coord,proj_coord)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    inp = torch.randn(1, 1, 256,256,2).cuda()
    feat=torch.randn(1, 64, 256,256,2).cuda()
    crop_hr=torch.randn(256,256,5)
    hr_coord, hr_value,proj_coord = to_pixel_samples(crop_hr,scale=(1,1,4))
    
    #sample_q=64*64*10
    #sample_lst = np.random.choice(hr_coord.shape[0], sample_q, replace=False)
    #hr_coord = hr_coord[sample_lst].unsqueeze(0).cuda()  # self.sample_q,3
    #proj_coord=proj_coord[sample_lst].unsqueeze(0).cuda()   #  self.sample_q,3
    
    hr_coord = hr_coord.unsqueeze(0).cuda()
    proj_coord=proj_coord.unsqueeze(0).cuda()
    model = SA_INR(add_res=True,add_branch=True).cuda()

    # calculate flops for each branch
    from thop import profile
    flops, params = profile(model, inputs=(inp,hr_coord,proj_coord))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    
    
    






