import os
import os.path as osp
import json
from argparse import ArgumentParser
from glob import glob

import torch
import cv2
from torch import cuda
from model import EAST
from tqdm import tqdm

from detect import detect
import utils


def simplify_ufo(ufo):
    # ufo format을 calc_deteval_metrics에 사용할 수 있도록 단순화
    ufo_simple = {}
    for image_name in ufo['images'].keys():
        words = ufo['images'][image_name]['words']
        boxes = []

        for _, value in words.items():
            points = value['points']
            boxes.append(points)
        ufo_simple[image_name] = boxes

    return ufo_simple

class Inference:
    def __init__(self, work_dir_path, conf, mode, device='cuda', input_size=2048, batch_size=1):
        self.model = None 
        self.work_dir_path = work_dir_path
        self.data_dir_path = conf['data_dir']
        self.json_dir_path = conf['json_dir']

        self.mode = mode
        self.device = device
        self.input_size = input_size
        self.batch_size = batch_size
        self.anns_simple = None

        languages = ['chinese', 'japanese', 'thai', 'vietnamese']
        
        # json 파일 이름 지정
        if mode == 'train':
            json_name = conf['train_json_name']
        elif mode == 'valid':
            json_name = conf['valid_json_name']
        elif mode == 'test':
            json_name = conf['test_json_name']

        # 이미지 경로 폴더
        if mode == 'train' or mode == 'valid':
            fold = 'train'
        elif mode == 'test':
            fold = 'test'

        total_image_names = []
        total_image_paths = []
        total_annos = {'images': {}}

        for language in languages:
            json_path = os.path.join(self.json_dir_path, f'{language}_receipt/ufo/{json_name}')
            json_data = utils.read_json(json_path)
            
            image_names = list(json_data['images'].keys())
            image_paths = []

            for image_name in image_names:
                image_paths.append(os.path.join(self.data_dir_path, f'{language}_receipt/img/{fold}/{image_name}'))
                total_annos['images'][image_name] = json_data['images'][image_name]

            total_image_names.extend(image_names)
            total_image_paths.extend(image_paths)

        if mode == 'train' or mode == 'valid':
            self.anns_simple = simplify_ufo(total_annos)

        self.image_names = total_image_names
        self.image_paths = total_image_paths
        self.anns = total_annos

        


    def inference(self, ckpt_path):
        if self.model is None:
            self.model = EAST(pretrained=False).to(self.device)

        model = self.model
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        with torch.no_grad():
            by_sample_bboxes, images, image_names = [], [], []
            
            for image_path in tqdm(self.image_paths):
                image_name = os.path.basename(image_path)
                image_names.append(image_name)

                images.append(cv2.imread(image_path)[:, :, ::-1])
                if len(images) == self.batch_size:
                    by_sample_bboxes.extend(detect(model, images, self.input_size))
                    images = []

            if len(images):
                by_sample_bboxes.extend(detect(model, images, self.input_size))

        ufo_result = dict(images=dict())
        for image_name, bboxes in zip(image_names, by_sample_bboxes):
            words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
            ufo_result['images'][image_name] = dict(words=words_info)

        return ufo_result

    def save_inference(self, ckpt_path, save_path=None, name_prefix=''):
        ckpt_path = os.path.join(self.work_dir_path, ckpt_path)
        
        if save_path is None:
            output_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            fold_path = os.path.join(self.work_dir_path, f'predicts/{self.mode}')
            save_path = os.path.join(fold_path, f'{name_prefix}_{self.mode}_{output_name}.csv')
        else:
            save_path = os.path.join(self.work_dir_path, save_path)
            fold_path = os.path.dirname(save_path)

        ufo_result = self.inference(ckpt_path)
        os.makedirs(fold_path, exist_ok=True)
        utils.save_json(ufo_result, save_path)
        return ufo_result



