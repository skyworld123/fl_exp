import argparse
import os
import torch

from utils.config import Config
from utils import check_test_model_path, ensure_dir, filter_files_with_ext, files_shallow, files_recursive, \
    set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        required=True,
                        type=str, help='Configuration file path.')
    parser.add_argument('-i', '--input',
                        required=True,
                        type=str, help='Path of the input data to be predicted. Can be a single '
                                       'file or a directory. If directory specified, predict all '
                                       'the inner files (depending on -r and -e args).')
    parser.add_argument('-r', '--recursive',
                        action='store_true',
                        help='Used when --input is a directory. If -r is specified, recursively '
                             'predict all files under this directory. If not, only predict shallow '
                             'files under the directory.')
    parser.add_argument('-e', '--ext',
                        default=None,
                        nargs='+', help='Allowed input file extension(s). If specified, the predictor '
                                        'ignores files with no extension. If not specified, predict '
                                        'any kinds of files (including the ones with no extension).')
    parser.add_argument('--model',
                        default=None,
                        type=str, help='Path of the pretrained model.')
    parser.add_argument('--seed',
                        default=32,
                        type=int, help='Random seed.')
    parser.add_argument('--device',
                        # default='cpu',
                        default='cuda:0',
                        type=str, help='Device used to predict.')
    parser.add_argument('--show_output',
                        action='store_true',
                        help='Whether to show output in real time when predicting.')
    parser.add_argument('-o', '--output_dir',
                        default='predict_results',
                        type=str, help='Directory to save the prediction results.')

    args = parser.parse_args()
    return args


def predict(cfg: Config):
    cfg_model = cfg.model
    cfg_predictor = cfg.predictor
    predictor_type = cfg_predictor['type']
    device = cfg.device
    show_output = cfg.args['show_output']
    output_dir = cfg.args['output_dir']
    transform = cfg.test_set['transform']

    input_path = cfg.args['input']
    rec = cfg.args['recursive']
    ext = cfg.args['ext']
    assert os.path.exists(input_path), f'Input path "{input_path}" does not exist.'
    if os.path.isfile(input_path):
        files = filter_files_with_ext([input_path], ext)
    elif os.path.isdir(input_path):
        if rec:
            files = files_recursive(input_path, ext)
        else:
            files = files_shallow(input_path, ext)
    else:
        raise NotImplementedError(f'Input path ({input_path}) should be a file or a directory. Other types '
                                  '(e.g. link) are not supported.')
    if len(files) == 0:
        print(f'No correct input file found (in {input_path}). Do not predict.')
        return

    model_path = cfg.args['model']
    pretrained_path = cfg_model['pretrained']
    model = cfg_model['type'](**cfg_model['params'])
    if model_path is None:
        model_path = pretrained_path
    if model_path is not None:
        print(f'Loading pretrained model {model_path}...')
        state_dict = torch.load(model_path, map_location=cfg.args['device'])
        model.load_state_dict(state_dict)
    else:
        print('[WARN] Model path not specified. Predict without loading pretrained model.')

    predictor = predictor_type(files, transform, model, device, show_output, output_dir,
                               **cfg_predictor['params'])

    predictor.predict()


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config(cfg_path)

    # check args
    args.model = check_test_model_path(cfg, args.model)
    if args.output_dir is not None:
        ensure_dir(args.output_dir)
    set_seed(args.seed)

    cfg.merge_args(args)

    predict(cfg)


if __name__ == '__main__':
    main()
