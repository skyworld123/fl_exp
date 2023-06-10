import argparse

from utils.config import Config
from utils import ensure_dir, default_output_dir, check_resume_dir, set_seed, save_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config',
                        required=True,
                        type=str, help='Configuration file path.')
    parser.add_argument('--output_dir',
                        default=None,
                        type=str, help='Directory to save output models and records.')
    parser.add_argument('--resume',
                        default=None,
                        type=str, help='Directory to resume training.')
    parser.add_argument('--eval',
                        action='store_true',
                        default=True,
                        help='Whether to evaluate during training.')
    parser.add_argument('--save_interval',
                        default=1,
                        type=int, help='How many iterations to save model checkpoint and records.')
    parser.add_argument('--keep_checkpoint_max',
                        default=5,
                        type=int, help='Maximum number of checkpoints to save.')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int, help='Number of workers for data loader.')
    parser.add_argument('--board',
                        action='store_true',
                        help='Whether to use Tensorboard to record the training process.')
    parser.add_argument('--seed',
                        default=32,
                        type=int, help='Random seed.')
    parser.add_argument('--device',
                        default='cuda:0',
                        type=str, help='Device used to train the model.')
    parser.add_argument('--verbose_train',
                        action='store_true',
                        help='Whether to output progress bar during training.')
    parser.add_argument('--verbose_val',
                        action='store_true',
                        help='Whether to output progress bar during evaluating.')
    parser.add_argument('--public_model',
                        default=1,
                        type=int, help='When set to >0, use a public model (and the corresponding public loss, '
                                       'optimizer and lr scheduler) for all clients. This prevent allocating an '
                                       'isolated model for each client, and therefore can save a great amount of '
                                       'memory in target device (e.g., GPU) when training with a large model & many '
                                       'clients.')

    args = parser.parse_args()
    return args


def train(cfg: Config):
    fl_method = cfg.fl_method
    manager = fl_method['type'](**fl_method['params'])

    manager.prepare(cfg)
    manager.train()
    manager.post_process()


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config(cfg_path)

    # check args
    if args.output_dir is None:
        args.output_dir = default_output_dir(cfg)
    ensure_dir(args.output_dir, remove_if_exist=args.resume is None)
    if args.resume is not None:
        check_resume_dir(args.resume)
    set_seed(args.seed)

    cfg.merge_args(args)
    save_config(cfg)

    train(cfg)


if __name__ == '__main__':
    main()
