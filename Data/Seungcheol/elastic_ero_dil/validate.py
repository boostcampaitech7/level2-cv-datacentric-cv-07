from inference import Inference, read_json, save_json, simplify_ufo
from deteval import calc_deteval_metrics

from glob import glob
import numpy as np
import os
import time

def sort(arr, indices):
    return [arr[index] for index in indices]

def get_trained_epochs(work_dir_path):
    trained_model_paths = glob(f'{work_dir_path}/trained_models/*.pth') # pth 파일 경로 얻기
    #trained_model_paths = glob(f'{work_dir_path}/trained_models/line_o_pre_line_x_json_add_aug/*.pth') # pth 파일 경로 얻기
    trained_model_paths = [path.replace(work_dir_path + '/', "") for path in trained_model_paths] # work_dir_path 제거
    
    trained_epochs = [int(os.path.splitext(os.path.basename(path))[0]) for path in trained_model_paths] # epoch 추출
    sorted_indices = np.argsort(trained_epochs)
    trained_epochs = sort(trained_epochs, sorted_indices)
    trained_model_paths = sort(trained_model_paths, sorted_indices)

    return trained_epochs, trained_model_paths

if __name__ == '__main__':
    mode = 'valid'
    save_name_prefix = 'data_split'

    work_dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir_path = os.path.join(work_dir_path, '../data_split')
    #save_predict_path_format = 'predictions/predicts/{mode}/{save_name_prefix}_{mode}_{epoch}.csv'
    save_predict_path_format = 'predicts/{mode}/{save_name_prefix}_{mode}_{epoch}.csv'
    #log_path = os.path.join(work_dir_path, f'predictions/predicts/{mode}/{mode}_log.txt')
    log_path = os.path.join(work_dir_path, f'predicts/{mode}/{mode}_log.txt')

    valid = Inference(data_dir_path, work_dir_path, mode='valid')

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, 'a')

    while True:
        time.sleep(1.0)
        trained_epochs, trained_model_paths = get_trained_epochs(work_dir_path)

        for epoch, ckpt_path in zip(trained_epochs, trained_model_paths):
            if epoch<39:
                continue
            save_predict_path = save_predict_path_format.format(mode=mode, save_name_prefix=save_name_prefix, epoch=epoch)
            if os.path.exists(save_predict_path):
                predicts = read_json(save_predict_path)
            else:
                predicts = valid.save_inference(ckpt_path, save_path=save_predict_path)
                
            print(f'Validating {save_predict_path}')
            predicts_simple = simplify_ufo(predicts)

            deteval = calc_deteval_metrics(predicts_simple, valid.anns_simple)
            precision, recall, hmean = deteval['total']['precision'], deteval['total']['recall'], deteval['total']['hmean']
            
            f.write(f'epoch: {epoch}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {hmean:.4f}\n')
            f.flush()
