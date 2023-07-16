import os
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
import random
import torch
import json
from gradient_loss import Get_gradient_loss
from utils import normalize, make_coord, to_pixel_samples, random_crop, crop_bg, center_crop


class MakeDataset(Dataset):

    def __init__(self,
                 path_list,
                 inp_size=(40, 40, 40),
                 scale_min=1,
                 scale_max=4,
                 sample_q=None,
                 is_train=True):
        self.path_list = path_list
        self.inp_size = inp_size
        self.inp_num = inp_size[0] * inp_size[1] * inp_size[2]
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.sample_q = sample_q
        self.is_train = is_train
        self.gradient_model = Get_gradient_loss()

    def __len__(self):
        return len(self.path_list)

    def sampling(self, volumn, scale):
        idxs0 = list(range(0, volumn.shape[0], scale[0]))
        idxs1 = list(range(0, volumn.shape[1], scale[1]))
        idxs2 = list(range(0, volumn.shape[2], scale[2]))
        sampled = volumn[idxs0, :, :][:, idxs1, :][:, :, idxs2]
        return sampled

    def get_mask(self, img, thre=0.8):

        inp = normalize(img)  # w*h*d
        inp = torch.FloatTensor(inp).view(1, 1, *inp.shape).cuda()

        gradient = self.gradient_model.get_gradient(inp)
        gradient = gradient.squeeze().cpu().numpy()

        threshold = np.quantile(gradient, thre)
        mask = (gradient > threshold).astype(np.int16)
        return mask

    def __getitem__(self, idx):

        img = sitk.GetArrayFromImage(sitk.ReadImage(self.path_list[idx]))
        img = img.T  # (10,256,256) -> (256,256,10)
        mask = self.get_mask(img)

        if self.is_train:

            s = random.randint(self.scale_min, self.scale_max)
            scale = (1, 1, s)
            hr_size = tuple(
                [scale[i] * (self.inp_size[i] - 1) + 1 for i in range(3)])
            res = random_crop(img, hr_size, mask)
        else:
            scale = (1, 1, 4)
            hr_size = tuple(
                [scale[i] * (self.inp_size[i] - 1) + 1 for i in range(3)])
            res = center_crop(img, hr_size, mask)
        crop_hr = res['crop']
        crop_mask = res['crop_mask']

        crop_hr = normalize(crop_hr)
        hr_coord, hr_value, proj_coord = to_pixel_samples(crop_hr, scale)

        if self.sample_q is not None:
            #print(hr_coord.shape,self.sample_q)
            sample_lst = np.random.choice(hr_coord.shape[0],
                                          self.sample_q,
                                          replace=False)
            hr_coord = hr_coord[sample_lst]  # self.sample_q,3
            hr_value = hr_value[sample_lst]  # self.sample_q,1
            proj_coord = proj_coord[sample_lst]  #  self.sample_q,3

        crop_lr = torch.FloatTensor(self.sampling(crop_hr, scale))
        sp = torch.FloatTensor([1]) - torch.sum(crop_lr) / self.inp_num
        #print(crop_lr.shape)

        return {
            'inp': crop_lr.unsqueeze(0),
            'coord': hr_coord,
            'proj_coord': proj_coord,
            #'gt': torch.FloatTensor(hr_value),
            'gt': torch.FloatTensor(crop_hr).unsqueeze(0),
            'crop_mask': crop_mask
        }


if __name__ == '__main__':
    with open("data.json", 'r') as f:
        f_dict = json.load(fp=f)
    f.close()
    dataset = MakeDataset(f_dict['train'], inp_size=(40, 40, 40))
    for i in range(len(dataset)):
        d = dataset[i]
        exit(0)