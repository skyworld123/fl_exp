import json
import os
import pickle
import torch
from typing import Union


class MessageSize:
    def __init__(self):
        self.buffer_class = {
            torch.nn.Module,
        }
        self.buffer = {}

    def message_size(self, obj) -> int:
        cls = type(obj)
        if cls in self.buffer_class:
            if cls not in self.buffer:
                b = pickle.dumps(obj)
                self.buffer[cls] = len(b)
            return self.buffer[cls]
        else:
            b = pickle.dumps(obj)
            return len(b)


def ensure_dir(d, remove_if_exist=False):
    if os.path.exists(d) and not os.path.isdir(d):
        raise RuntimeError(f'Expect to create a directory {d}, but the path, which already exists, is not a directory.')
    if os.path.exists(d) and remove_if_exist:
        remove_dir_recursive(d)
    if not os.path.exists(d):
        os.makedirs(d)


def remove_dir_recursive(d):
    if not os.path.exists(d):
        raise FileNotFoundError(f'Directory {d} not found.')
    if os.path.exists(d) and not os.path.isdir(d):
        raise RuntimeError(f'Expect to recursively remove directory {d}, but the path is not a directory.')

    for root, dirs, files in os.walk(d, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(d)


def default_output_dir(cfg):
    from .config import Config

    assert isinstance(cfg, Config)
    output_root_dir = 'output'
    fl_method = cfg.dic['fl_method']['type']
    dataset = cfg.dic['train_set']['type']
    model = cfg.dic['model']['type']
    iid_info = cfg.dic['train_set']['division']['method']
    if fl_method == 'Centralized':
        subdir = f'{fl_method}_{dataset}_{model}'
    else:
        subdir = f'{fl_method}_{dataset}_{model}_{iid_info}'
    output_dir = os.path.join(output_root_dir, subdir)
    return output_dir


def output_epoch_dir(output_dir, epoch: int):
    epoch_dir = f'epoch{epoch + 1}'
    return os.path.join(output_dir, epoch_dir)


def output_best_dir(output_dir):
    return os.path.join(output_dir, 'best_model')


def check_resume_dir(resume_dir):
    if not os.path.exists(resume_dir):
        raise FileNotFoundError(f'Resume directory {resume_dir} not found.')
    # file_name_list = ['client_state', 'random']
    # resume_list = [x + '.pth' for x in file_name_list]
    # for file in resume_list:
    #     file_path = os.path.join(resume_dir, file)
    #     if not os.path.exists(file_path):
    #         raise FileNotFoundError(f'Resume file {file} not found in {resume_dir}.')


def check_test_model_path(cfg, model_path):
    if model_path is None:
        # assume the best model under default output directory
        output_dir = default_output_dir(cfg)
        model_path = os.path.join(output_best_dir(output_dir), 'model.pth')
        if not os.path.exists(model_path):
            print(f'WARNING: Model path not specified in args. Assumed model path {model_path} not found.')
            return None
        return model_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model path {model_path} not found.')
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Input model path is a directory. Assumed model path {model_path} not found.')
    return model_path


def load_resume(resume_dir, obj_type: str):
    """
    Load the state dict of the object (client state, server state, random state or info) to resume.
    """
    assert obj_type in ['model', 'client_state', 'server_state', 'random', 'info']

    file = f'{obj_type}.pth'
    file_path = os.path.join(resume_dir, file)
    if not os.path.exists(file_path):
        return None
    state_dict = torch.load(file_path)
    return state_dict


def save_ckpt(checkpoint_dir, obj_type: str, obj):
    """
    Save checkpoint object (client state, server state, random state or info).
    """
    assert obj_type in ['model', 'client_state', 'server_state', 'random', 'info']

    file = f'{obj_type}.pth'
    file_path = os.path.join(checkpoint_dir, file)
    torch.save(obj, file_path)


def save_ckpt_multiple(checkpoint_dir, obj_dict: dict):
    """
    Save multiple checkpoint objects.
    obj_dict: (e.g.) {'client_state': client_state, 'server_state': server_state, ...}
    """
    obj_type_list = ['model', 'client_state', 'server_state', 'random', 'info']
    for obj_type, obj in obj_dict.items():
        if obj_type in obj_type_list:
            save_ckpt(checkpoint_dir, obj_type, obj)


def save_config(cfg):
    output_dir = cfg.args['output_dir']
    config_file = 'config.json'
    output_path = os.path.join(output_dir, config_file)

    f = open(output_path, 'w')
    json.dump(cfg.dic, f, indent=2)
    f.close()


def filter_files_with_ext(files: list, ext: Union[str, list] = None):
    if ext is None or (isinstance(ext, list) and len(ext) == 0):
        return files
    if isinstance(ext, str):
        ext = [ext]
    ext = [e.split('.')[-1] for e in ext]

    filtered = []
    for file in files:
        file_name = file.replace('\\', '/').split('/')[-1]
        file_ext = file_name.split('.')[-1]
        if file_ext in ext:
            filtered.append(file)

    return filtered


def files_shallow(d: str, ext: Union[str, list] = None):
    assert os.path.isdir(d)
    files = [os.path.join(d, file) for file in os.listdir(d) if os.path.isfile(os.path.join(d, file))]
    return filter_files_with_ext(files, ext)


def files_recursive(d: str, ext: Union[str, list] = None):
    assert os.path.isdir(d)
    files = []
    for root, dirs, fs in os.walk(d):
        files += [os.path.join(root, f) for f in fs]
    return filter_files_with_ext(files, ext)


def save_data_in_csv(data: list, save_path):
    f = open(save_path, 'w', encoding='utf-8')

    data_raw = ''
    for line in data:
        assert isinstance(line, list)
        line_str = []
        for item in line:
            item = str(item)
            if item.find(',') != -1:
                item = f'"{item}"'
            line_str.append(item)
        line_str = ','.join(line_str) + '\n'
        data_raw += line_str
    f.write(data_raw)

    f.close()

