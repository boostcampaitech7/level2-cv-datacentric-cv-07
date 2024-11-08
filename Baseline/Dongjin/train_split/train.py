import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import utils


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp_path', type=str, default=None) # 
    args = parser.parse_args()
    return args


def do_training(conf):
    dataset = SceneTextDataset(
        conf['data_dir'],
        split='train',
        image_size=conf['image_size'],
        crop_size=conf['input_size'],
        json_name=conf['train_json_name'],
        json_dir=conf['json_dir']
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / conf['batch_size'])
    train_loader = DataLoader(
        dataset,
        batch_size=conf['batch_size'],
        shuffle=True,
        num_workers=conf['num_workers']
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)

    if conf['pretrained_path'] is not None:
        model.load_state_dict(torch.load(conf['pretrained_path']))

    optimizer = torch.optim.Adam(model.parameters(), lr=conf['learning_rate'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[conf['max_epoch'] // 2], gamma=0.1)

    model.train()
    for epoch in range(conf['max_epoch']):
        epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

        if (epoch + 1) % conf['save_interval'] == 0:
            if not osp.exists(conf['model_dir']):
                os.makedirs(conf['model_dir'])

            ckpt_fpath = osp.join(conf['model_dir'], f'{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)



if __name__ == '__main__':    
    work_dir_path = os.path.dirname(os.path.realpath(__file__))

    args = parse_args()
    exp_path = args.exp_path

    conf = utils.load_conf(work_dir_path=work_dir_path, rel_exp_path=exp_path)
    conf['work_dir_path'] = work_dir_path

    if conf['input_size'] % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    conf['model_dir'] = os.path.join(work_dir_path, f"trained_models/{conf['run_name']}")
    conf['model_dir'] = utils.renew_if_path_exist(conf['model_dir'])
    conf['run_name'] = os.path.basename(conf['model_dir'])

    save_conf_path = os.path.join(conf['model_dir'], 'exp.json')
    os.makedirs(conf['model_dir'], exist_ok=True)
    utils.save_json(conf, save_conf_path)

    do_training(conf)