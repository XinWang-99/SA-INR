import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import os
from utils import normalize

class Get_gradient_loss(nn.Module):
    def __init__(self):
        super(Get_gradient_loss, self).__init__()
        
        kernel_w,kernel_h,kernel_d = torch.zeros((3,3,3)),torch.zeros((3,3,3)),torch.zeros((3,3,3))
        
        kernel_w[0,1,1]=-1
        kernel_w[2,1,1]=1
        
        kernel_h[1,1,0]=-1
        kernel_h[1,1,2]=1
        
        kernel_d[1,0,1]=-1
        kernel_d[1,2,1]=1
       

        self.weight_w = nn.Parameter(data = kernel_w, requires_grad = False).view(1,1,3,3,3).cuda()
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).view(1,1,3,3,3).cuda()
        self.weight_d = nn.Parameter(data = kernel_d, requires_grad = False).view(1,1,3,3,3).cuda()

        self.L1loss = nn.L1Loss()

    def get_gradient(self, x):  # x: (b,1,w,h,d)  k: (1,1,3,3,3)  output: (b,1,w,h,d)
    
        #print(x.shape)
        g_w=F.conv3d(x,self.weight_w,padding='valid' )
        g_h=F.conv3d(x,self.weight_h,padding='valid' )
        g_d=F.conv3d(x,self.weight_d,padding='valid' )
        
        output = torch.sqrt(torch.pow(g_w, 2) + torch.pow(g_h, 2)+ torch.pow(g_d, 2)+ 1e-6)
        p3d = (1,1,1,1,1,1)
        output=F.pad(output,p3d,mode='replicate')
        return output
    
    def get_mask(self,gradient,thre=0.8):
        
        threshold=torch.quantile(gradient,torch.tensor(thre).cuda())
        mask=(gradient>threshold).float()
        return mask
        
    def expand_mask(self,mask):
        
        dilated_kernel=torch.ones((1,1,3,3,3)).cuda()
        expanded_mask=F.conv3d(mask,dilated_kernel,padding='valid' )
        p3d = (1,1,1,1,1,1)
        expanded_mask=F.pad(expanded_mask,p3d,mode='replicate')
        return expanded_mask
            
    def forward(self,x1,x2):
        
        gradient_1=self.get_gradient(x1)
        gradient_2=self.get_gradient(x2)
        
        gradient_loss=self.L1loss(gradient_1,gradient_2)

        
        return gradient_loss
        
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    a=Get_gradient_loss()
    img_dir="/mnt/shared_storage/wangxin/skull_stripped/"
    for f in os.listdir(img_dir):
       
        path=os.path.join(img_dir,f)
        print(path)
        Image=sitk.ReadImage(path)   # LR image

        img = sitk.GetArrayFromImage(Image)
     
    
        inp = normalize(img) # w*h*d 
        inp=torch.FloatTensor(inp).view(1,1,*inp.shape).cuda()
        gradient=a.get_gradient(inp)
        gradient_=gradient.squeeze().cpu().numpy()
    
        gradient_img=sitk.GetImageFromArray(gradient_)
        gradient_img.SetSpacing(Image.GetSpacing())
        gradient_img.SetOrigin(Image.GetOrigin())
        gradient_img.SetDirection(Image.GetDirection())
        sitk.WriteImage(gradient_img,os.path.join(img_dir,f,'gradient.nii.gz'))
        
        #mask=a.get_mask(gradient).squeeze().cpu().numpy()
        #mask_img=sitk.GetImageFromArray(mask)
        #mask_img.SetSpacing(Image.GetSpacing())
        #mask_img.SetOrigin(Image.GetOrigin())
        #sitk.WriteImage(mask_img,'branch/00f52d59-f1a8969c-63d54808-0c04bed5-d66035fe/1/gradient_mask.nii.gz')
        #exit(0)
