import torch
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import json
import argparse
import os
import yaml
import utils
from network.model import SA_INR
from make_dataset import MakeDataset
from gradient_loss import Get_gradient_loss


def make_data_loader(f_dict, tag=''):

    if tag == 'train':
        is_train = True
    else:
        is_train = False

    dataset = MakeDataset(f_dict[tag], is_train=is_train)

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset,
                        batch_size=config.get('batch_size'),
                        shuffle=False,
                        num_workers=8,
                        pin_memory=True)
    return loader


def make_data_loaders():
    with open(config.get('data'), 'r') as f:
        f_dict = json.load(fp=f)
    f.close()
    train_loader = make_data_loader(f_dict, tag='train')
    val_loader = make_data_loader(f_dict, tag='val')
    return train_loader, val_loader


def prepare_training():
    print("add_res: {}".format(args.add_res),
          "add_NLSA: {}".format(args.add_NLSA),
          "add_branch: {}".format(args.add_branch),
          "layerType: {}".format(args.layerType),
          "dilation: {}".format(args.dilation))
    model = SA_INR(layerType=args.layerType,
                   dilation=args.dilation,
                   add_res=args.add_res,
                   add_NLSA=args.add_NLSA,
                   add_branch=args.add_branch).cuda()

    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    epoch_start = 1
    if args.preTrain_path != None:
        st = torch.load(args.preTrain_path, map_location='cpu')
        model.load_state_dict(st['model'])
        optimizer.load_state_dict(st['optimizer'])
        epoch_start = 2000
    if config.get('multi_step_lr') is None:
        lr_scheduler = None
    else:
        lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer, gamma):
    model.train()
    loss_fn = nn.L1Loss()
    G_loss_model = Get_gradient_loss().cuda()
    train_loss = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):

        for k, v in batch.items():
            batch[k] = v.cuda()

        output = model(batch['inp'], batch['coord'], batch['proj_coord'])
        #print(pred.shape,batch['gt'].shape)
        output['pred'] = output['pred'].view(batch['gt'].shape)  # (1,1,w,h,d)
        output['mask'] = output['mask'].view(batch['crop_mask'].shape)

        loss = loss_fn(output['pred'], batch['gt'])
        if args.add_branch:
            sparsity = torch.sum(
                batch['crop_mask']) / batch['crop_mask'].view(-1).shape[0]
            pred = torch.sum(output['mask']) / output['mask'].view(-1).shape[0]
            #print(sparsity,pred)
            loss = loss + gamma * loss_fn(
                output['mask'], batch['crop_mask']) + (sparsity - pred)**2

        if args.gradient_loss:
            g_loss = G_loss_model(output['pred'], batch['gt'])
            #loss=loss+g_loss/(g_loss/loss_base).detach()
            loss = loss + 0.1 * g_loss
        #print(loss)
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss.item()


@torch.no_grad()
def eval_psnr(eval_loader, model):
    model.eval()
    val_res = utils.Averager()
    for batch in tqdm(eval_loader, leave=False, desc='eval'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        output = model(batch['inp'], batch['coord'], batch['proj_coord'])
        output['pred'] = output['pred'].view(batch['gt'].shape)  # (1,1,w,h,d)
        output['pred'].clamp_(0, 1)

        mse = (output['pred'] - batch['gt']).pow(2).mean()
        psnr = -10 * torch.log10(mse)
        val_res.add(psnr.item(), batch['inp'].shape[0])
    return val_res.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        print("model parallel")
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    gamma = 1
    for epoch in range(epoch_start, epoch_max + 1):
        if (epoch + 1) % 50 == 0:
            gamma = gamma / 2

        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, gamma)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}, gamma={}'.format(
            train_loss, gamma))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        sv_file = {
            'model': model_.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1:
                model_ = model.module
            else:
                model_ = model
            # #
            val_res = eval_psnr(val_loader, model_)

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='train_SA_INR.yaml')
    parser.add_argument('--save_path', default='base_new')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--add_res', action='store_true')
    parser.add_argument('--add_NLSA', action='store_true')
    parser.add_argument('--add_branch', action='store_true')
    parser.add_argument('--gradient_loss', action='store_true')
    parser.add_argument('--layerType', default='FBLA')
    parser.add_argument('--dilation', default=2, type=int)
    parser.add_argument('--preTrain_path', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    save_path = os.path.join(args.save_path,
                             '_' + args.config.split('/')[-1][:-len('.yaml')])
    mp.set_start_method('spawn')
    main(config, save_path)
