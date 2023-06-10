import argparse
import torch

from flmethods.tools import generate_test_dataloader_default
from utils.config import Config
from utils import check_test_model_path, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        required=True,
                        type=str, help='Configuration file path.')
    parser.add_argument('--model',
                        default=None,
                        type=str, help='Path of the pretrained model.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int, help='Number of workers for data loader.')
    parser.add_argument('--seed',
                        default=32,
                        type=int, help='Random seed.')
    parser.add_argument('--device',
                        default='cuda:0',
                        type=str, help='Device used to test the model.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Whether to output progress bar during testing.')

    args = parser.parse_args()
    return args


def test(cfg: Config):
    dataloader = generate_test_dataloader_default(cfg)

    cfg_model = cfg.model
    cfg_tester = cfg.tester
    tester_type = cfg_tester['type']
    device = cfg.device
    verbose = cfg.args['verbose']

    model_path = cfg.args['model']
    pretrained_path = cfg_model['pretrained']
    model = cfg_model['type'](**cfg_model['params'])
    if model_path is None:
        model_path = pretrained_path
    if model_path is not None:
        print(f'Loading pretrained model {model_path}...')
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
    else:
        print('[WARN] Model path not specified. Test without loading pretrained model.')

    tester = tester_type(dataloader, model, device,
                         verbose=verbose,
                         **cfg_tester['params'])

    tester.test()


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config(cfg_path)

    # check args
    args.model = check_test_model_path(cfg, args.model)
    set_seed(args.seed)

    cfg.merge_args(args)

    test(cfg)


if __name__ == '__main__':
    main()
