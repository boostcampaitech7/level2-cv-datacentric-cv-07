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
    # train.py 실행 시, 실험 조건 파일(exp.json)과 모델 가중치 파일(.pth)가 trained_models/run_name 폴더에 저장합니다.
    # validate.py는 저장된 exp.json과 모델 가중치 파일을 불러와서 valid set(valid_json_name)을 추론합니다.
    # 사용 예시: python validate.py

    mode = 'valid'
    work_dir_path = os.path.dirname(os.path.realpath(__file__)) # 현재 python 파일 디렉토리
    save_predict_path_format = 'predicts/{run_name}/{mode}/{run_name}_{mode}_{epoch}.csv' # 예상 결과 저장할 경로 형식 지정
    log_path_format = os.path.join(work_dir_path, 'predicts/{run_name}/{mode}/{mode}_log.txt') # 로그 저장할 경로 형식 지정

    while True:
        time.sleep(1)
        # trained_models 폴더 안에 exp.json 파일 찾기
        conf_paths = glob(work_dir_path + '/trained_models/*/exp.json', recursive=True) 
        
        for conf_path in conf_paths: 
            conf_dir_path = os.path.dirname(conf_path) # 결과 저장 디렉토리
            conf = utils.read_json(conf_path) # 실험 설정 불러오기 
            run_name = conf['run_name'] # run_name 읽어오기

            # 결과 저장 디렉토리에 존재하는 실험 epoch과 pth weight path 불러오기
            trained_epochs, trained_model_paths = get_trained_epochs(conf_dir_path) 
            trained_model_paths = rm_work_dir_paths(work_dir_path, trained_model_paths)
            
            # 로그 경로 설정
            log_path = log_path_format.format(run_name=run_name, mode=mode)
            log_lock_path = log_path + '.lock' # log 기록 시 다른 프로세스 접근 대기 설정
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            valid = None # 모델 설정 초기화

            for epoch, ckpt_path in zip(trained_epochs, trained_model_paths):
                # 추론결과 저장경로 설정
                save_predict_path = os.path.join(work_dir_path,
                                                 save_predict_path_format.format(run_name=run_name, mode=mode, epoch=epoch))
                
                # 추론결과가 존재하면 다음 루프로 넘어감 
                if os.path.exists(save_predict_path):
                    continue
                
                # 빈 파일 생성: 프로세스 동시 실행 시 작업 안겹치는 용도
                with open(save_predict_path, 'a') as f:
                    f.write("")

                # validation 시작 
                print(f'Validating {save_predict_path}')

                # 모델 불러오기 
                if valid is None:
                    print(f"load model with {conf_path}")
                    valid = Inference(work_dir_path=work_dir_path, conf=conf, mode=mode)

                # 추론 시작 
                predicts = valid.save_inference(ckpt_path, save_path=save_predict_path)
                predicts_simple = simplify_ufo(predicts) # calc_deteval_metrics을 위한 라벨 단순화

                # validation metric 측정
                deteval = calc_deteval_metrics(predicts_simple, valid.anns_simple)
                precision, recall, hmean = deteval['total']['precision'], deteval['total']['recall'], deteval['total']['hmean']
                
                # 로그 기록 및 저장
                log = f'epoch: {epoch}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {hmean:.4f}'
                print(log)

                # 로그 저장
                with FileLock(log_lock_path): # 다른 프로세스 접근 방지
                    # 로그 파일이 존재하면 추가
                    if os.path.exists(log_path):
                        with open(log_path, 'r') as f:
                            logs = f.readlines()
                        logs.append(log) # 로그 추가

                        # 로그 정렬
                        logs_dict = utils.read_log(logs=logs)
                        indices = np.argsort(logs_dict['epoch'])
                        logs = [logs[index].replace("\n", "")+"\n" for index in indices]

                        # 로그 저장
                        with open(log_path, 'w') as f:
                            f.writelines(logs) 

                    # 로그 파일이 존재하지 않으면 생성 및 저장
                    else:
                        with open(log_path, 'w') as f:
                            f.write(log + '\n')


