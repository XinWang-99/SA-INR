import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from network.model import SA_INR
from utils import input_matrix_wpn, make_coord_2, normalize


def test(model, path, slice_spacing, save=False):
    print(path)

    Image = sitk.ReadImage(path)  # LR image
    print(Image.GetSize())

    spacing = Image.GetSpacing()  # (x,x,3.3/4.5)
    print("spacing: ", spacing)

    target_spacing = (spacing[0], spacing[0], slice_spacing)
    spacing_ratio = (1, 1, slice_spacing / target_spacing[2])

    img = sitk.GetArrayFromImage(Image).T
    inp = normalize(img).astype(np.float64)  # w*h*d
    lr = copy.deepcopy(inp)

    shape = (inp.shape[0], inp.shape[1],
             int((inp.shape[2] - 1) * spacing_ratio[2]) + 1)
    print(shape)
    hr_coord = make_coord_2(inp.shape,
                            shape,
                            spacing_ratio,
                            ranges=None,
                            flatten=False)  # (w,h,d,3)
    proj_coord = input_matrix_wpn(*shape, scale=spacing_ratio, flatten=False)

    inp = torch.FloatTensor(inp).view(1, 1, *inp.shape).cuda()
    feat = model.get_feat(inp)
    sr = []
    for i in range(shape[2]):
        hr_coord_d = hr_coord[:, :, i, :].view(-1, 3).unsqueeze(0).cuda()
        proj_coord_d = proj_coord[:, :, i, :].view(-1, 3).unsqueeze(0).cuda()
        output = model.inference(inp, feat, hr_coord_d, proj_coord_d)
        pred = output['pred'].view(shape[0],
                                   shape[1]).cpu().numpy().astype(np.float64)
        sr.append(pred)
    sr = np.clip(np.stack(sr, axis=-1), 0, 1)

    if save:

        save_dir = os.path.join(args.save_dir,
                                path.split('/')[-2], str(target_spacing[2]))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sr_path = '{}/sr.nii.gz'.format(save_dir)
        sr_img = sitk.GetImageFromArray(sr.T)
        sr_img.SetSpacing(target_spacing)
        sr_img.SetOrigin(Image.GetOrigin())
        sr_img.SetDirection(Image.GetDirection())
        sitk.WriteImage(sr_img, sr_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--add_res', action='store_true')
    parser.add_argument('--add_NLSA', action='store_true')
    parser.add_argument('--add_branch', action='store_true')
    parser.add_argument('--layerType', default='FBLA')
    parser.add_argument('--dilation', default=2, type=int)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--slice_spacing', default=1, type=float)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("add_res: {}".format(args.add_res),
          "add_NLSA: {}".format(args.add_NLSA),
          "add_branch: {}".format(args.add_branch),
          "layerType: {}".format(args.layerType),
          "dilation: {}".format(args.dilation))
    model = SA_INR(layerType=args.layerType,
                   dilation=args.dilation,
                   add_res=args.add_res,
                   add_NLSA=args.add_NLSA,
                   add_branch=args.add_branch,
                   is_train=False)
    st = torch.load(args.model_path, map_location='cpu')['model']
    model.load_state_dict(st)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    print("model loaded")

    with open("clinical_knee.json", 'r') as f:
        f_dict = json.load(fp=f)

    for path in f_dict['val']:
        path = path.replace('xr', 'x')
        with torch.no_grad():
            sp = test(model, path, args.slice_spacing, save=True)
        exit(0)
