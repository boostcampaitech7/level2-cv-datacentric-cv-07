import os
import json
import re

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def save_json(dicts, file_path):
    with open(file_path, "w") as f:
        json.dump(dicts, f, indent=4)

def load_conf(work_dir_path='', rel_default_path='conf/default.json', rel_exp_path=None):
    default_path = os.path.join(work_dir_path, rel_default_path)
    conf = read_json(default_path)

    if rel_exp_path is not None:
        exp_path = os.path.join(work_dir_path, rel_exp_path)
        new_conf = read_json(exp_path)

        for key, value in new_conf.items():
            conf[key] = value

    return conf


def renew_if_path_exist(path):
    if not(os.path.exists(path)):
        return path
    else:
        for i in range(100000):
            temp_path = path + f'_{i}'
            if not(os.path.exists(temp_path)):
                return temp_path            
                
        raise Exception("path 지정을 실패했습니다.")

def read_log(log_path):
    with open(log_path, 'r') as f:
        logs = f.readlines()

    result = {'epoch': [], 'precision': [], 'recall': [], 'f1': []}

    for log in logs:
        match = re.search(r'epoch: (\d+), precision: ([0-9.]+), recall: ([0-9.]+), f1: ([0-9.]+)', log)
        result['epoch'].append(int(match.group(1)))
        result['precision'].append(float(match.group(2)))
        result['recall'].append(float(match.group(3)))
        result['f1'].append(float(match.group(4)))

    return result

def read_log(log_path=None, logs=None):
    if logs is None:
        with open(log_path, 'r') as f:
            logs = f.readlines()

    result = {'epoch': [], 'precision': [], 'recall': [], 'f1': []}

    for log in logs:
        match = re.search(r'epoch: (\d+), precision: ([0-9.]+), recall: ([0-9.]+), f1: ([0-9.]+)', log)
        result['epoch'].append(int(match.group(1)))
        result['precision'].append(float(match.group(2)))
        result['recall'].append(float(match.group(3)))
        result['f1'].append(float(match.group(4)))

    return result

