from glob import glob
import os
import re
import numpy as np
import utils
from inference import Inference


if __name__ == '__main__':
    mode = 'test'
    best_metric = 'f1'

    test_path_format = 'predicts/{run_name}/{mode}/{run_name}_{mode}_ep_{epoch}_f1_{f1:.4f}.csv'
    valid_log_path_format = 'predicts/{run_name}/valid/valid_log.txt'
    pth_path_format = 'trained_models/{run_name}/{epoch}.pth'

    work_dir_path = os.path.dirname(os.path.realpath(__file__)) # 현재 python 파일 디렉토리
    conf_paths = glob(work_dir_path + '/trained_models/*/exp.json', recursive=True) 
    conf_paths.sort()

    for conf_path in conf_paths: 
        conf_dir_path = os.path.dirname(conf_path)
        conf = utils.read_json(conf_path)
        run_name = conf['run_name']

        valid_log_path = os.path.join(work_dir_path, valid_log_path_format.format(run_name=run_name))
        valid_log = utils.read_log(valid_log_path)

        select = {}
        select_index = np.argmax(valid_log[best_metric])
        select['epoch'] = valid_log['epoch'][select_index]
        select['precision'] = valid_log['precision'][select_index]
        select['recall'] = valid_log['recall'][select_index]
        select['f1'] = valid_log['f1'][select_index]

        ckpt_path = os.path.join(work_dir_path, 
                                pth_path_format.format(run_name=run_name, epoch=select['epoch']))
        save_test_path = os.path.join(work_dir_path,
                                test_path_format.format(run_name=run_name, mode=mode, epoch=select['epoch'], f1=select['f1']))
        
        if os.path.exists(save_test_path):
            continue

        # test 빈 파일 만들기 (중복 작업 방지)
        os.makedirs(os.path.dirname(save_test_path), exist_ok=True)
        with open(save_test_path, 'w') as f:
            f.write("")
        
        print(f"Predicting and saving {save_test_path}...")
        test = Inference(work_dir_path=work_dir_path, conf=conf, mode=mode)
        predicts = test.save_inference(ckpt_path, save_path=save_test_path)
