from inference import Inference, simplify_ufo
from deteval import calc_deteval_metrics

from glob import glob
import numpy as np
import os
import time
import utils
from filelock import FileLock
import numpy as np


def rm_work_dir_paths(work_dir_path, paths):
    return [path.replace(work_dir_path + '/', "") for path in paths]

def sort(arr, indices):
    return [arr[index] for index in indices]

def get_trained_epochs(conf_dir_path):
    trained_model_paths = glob(f'{conf_dir_path}/*.pth') # pth 파일 경로 얻기
        
    trained_epochs = [int(os.path.splitext(os.path.basename(path))[0]) for path in trained_model_paths] # epoch 추출
    sorted_indices = np.argsort(trained_epochs)
    trained_epochs = sort(trained_epochs, sorted_indices)
    trained_model_paths = sort(trained_model_paths, sorted_indices)

    return trained_epochs, trained_model_paths


if __name__ == '__main__':
    mode = 'valid'
    work_dir_path = os.path.dirname(os.path.realpath(__file__)) # 현재 python 파일 디렉토리
    save_predict_path_format = 'predicts/{run_name}/{mode}/{run_name}_{mode}_{epoch}.csv'
    log_path_format = os.path.join(work_dir_path, 'predicts/{run_name}/{mode}/{mode}_log.txt')
    

    while True:
        # trained_models 폴더 안에 exp.json 파일 찾기
        time.sleep(1)
        conf_paths = glob(work_dir_path + '/trained_models/*/exp.json', recursive=True) 
        
        for conf_path in conf_paths: 
            conf_dir_path = os.path.dirname(conf_path)
            conf = utils.read_json(conf_path)
            run_name = conf['run_name']

            trained_epochs, trained_model_paths = get_trained_epochs(conf_dir_path)
            trained_model_paths = rm_work_dir_paths(work_dir_path, trained_model_paths)
            
            log_path = log_path_format.format(run_name=run_name, mode=mode)
            log_lock_path = log_path + '.lock'
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            valid = None # 모델 제거, conf 파일이 바뀔 때 마다 Inference 초기값이 달라져야 함

            for epoch, ckpt_path in zip(trained_epochs, trained_model_paths):
                save_predict_path = os.path.join(work_dir_path,
                                                 save_predict_path_format.format(run_name=run_name, mode=mode, epoch=epoch))
                
                if os.path.exists(save_predict_path):
                    continue
                
                # 빈 파일 만들어서 작업 안겹치게 하기
                with open(save_predict_path, 'a') as f:
                    f.write("")

            
                print(f'Validating {save_predict_path}')
                if valid is None:
                    print(f"load model with {conf_path}")
                    valid = Inference(work_dir_path=work_dir_path, conf=conf, mode=mode)

                predicts = valid.save_inference(ckpt_path, save_path=save_predict_path)
                predicts_simple = simplify_ufo(predicts)

                deteval = calc_deteval_metrics(predicts_simple, valid.anns_simple)
                precision, recall, hmean = deteval['total']['precision'], deteval['total']['recall'], deteval['total']['hmean']
                
                log = f'epoch: {epoch}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {hmean:.4f}'
                print(log)

                # 로그 기록하기
                with FileLock(log_lock_path):
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            logs = f.readlines()
                        logs.append(log)

                        logs_dict = utils.read_log(logs=logs)
                        indices = np.argsort(logs_dict['epoch'])
                        logs = [logs[index].replace("\n", "")+"\n" for index in indices]

                        with open(log_path, 'w') as f:
                            f.writelines(logs)
                    else:
                        with open(log_path, 'w') as f:
                            f.write(log + '\n')

