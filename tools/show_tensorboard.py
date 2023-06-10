import argparse
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--logdir',
                        required=True,
                        type=str, help='logdir')
    parser.add_argument('-k', '--key',
                        required=True,
                        type=str, help='key, also y label')
    parser.add_argument('-s', '--steps',
                        default='(0, 100)',
                        type=str, help='step range')
    parser.add_argument('--size',
                        default='(12, 8)',
                        type=str, help='figure size')
    parser.add_argument('--xlabel',
                        default='epoch',
                        type=str, help='x label')
    parser.add_argument('--legend_loc',
                        default='lower right',
                        choices=['upper left', 'upper right', 'lower left', 'lower right'],
                        type=str, help='location of legends')
    parser.add_argument('-o', '--output_dir',
                        default=None,
                        type=str, help='output dir')
    parser.add_argument('-n', '--output_name',
                        default=None,
                        type=str, help='output name')
    parser.add_argument('-e', '--ext',
                        default='png',
                        type=str, help='output picture extension')
    parser.add_argument('--dpi',
                        default=300,
                        type=int, help='dpi of the output picture')

    args = parser.parse_args()
    return args


def ensure_dir(d):
    if os.path.exists(d) and not os.path.isdir(d):
        raise RuntimeError(f'Expect to create a directory {d}, but the path, which already exists, is not a directory.')
    if not os.path.exists(d):
        os.makedirs(d)


def find_log_paths(logdir):
    subdirs = os.listdir(logdir)
    subdirs = [d for d in subdirs if os.path.isdir(os.path.join(logdir, d))]

    log_paths, labels = [], []
    for d in subdirs:
        d_path = os.path.join(logdir, d)
        files = [f for f in os.listdir(d_path) if f.startswith('events.out.tfevents')]
        if len(files) == 0:
            continue
        file = files[0]
        if len(files) > 1:
            print(f'warning: found {len(files)} tfevents files in {d_path}, only process {file}')
        log_path = os.path.join(d_path, file)
        log_paths.append(log_path)
        labels.append(d)

    return log_paths, labels


def draw_data(log_paths: list[str],
              labels: list[str],
              key: str,
              output_dir: str,
              output_name: str,
              ext: str = '.png',
              dpi: int = 300,
              **kwargs):
    ext = '.' + ext.split('.')[-1]
    assert ext in ['.jpg', '.png', '.jpeg', '.svg', '.tif'], f'unsupported format: "{ext}"'
    assert dpi > 0, f'dpi should be >0, got {dpi}'

    steps = kwargs['steps']
    size = kwargs['size']
    xlabel = kwargs['xlabel']
    legend_loc = kwargs['legend_loc']

    fig = plt.figure(figsize=size)

    ax1 = fig.add_subplot()
    ax1.set_xlim(steps[0], steps[1])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(key)

    for log_path, label in zip(log_paths, labels):
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        if key in ea.scalars.Keys():
            values = ea.scalars.Items(key)
            ax1.plot([i.step for i in values], [i.value for i in values], label=label)

    plt.legend(loc=legend_loc)

    output_path = os.path.join(output_dir, f'{output_name}{ext}')
    plt.savefig(output_path, dpi=dpi)
    print(f'Tensorboard result saved in {output_path}.')


def main():
    args = parse_args()

    assert os.path.exists(args.logdir), f'dir {args.logdir} not exist'
    assert os.path.isdir(args.logdir), f'path {args.logdir} is not a directory'
    log_paths, labels = find_log_paths(args.logdir)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = '../tensorboard_results'
    ensure_dir(output_dir)
    output_name = args.output_name
    if output_name is None:
        output_name = f'result-{args.key}'

    # kwargs
    steps = eval(args.steps)
    assert len(steps) == 2 and steps[0] < steps[1]
    size = eval(args.size)
    assert len(size) == 2 and size[0] > 0 and size[1] > 0
    xlabel = args.xlabel
    legend_loc = args.legend_loc
    kwargs = {
        'steps': steps,
        'size': size,
        'xlabel': xlabel,
        'legend_loc': legend_loc,
    }

    draw_data(log_paths, labels, args.key,
              output_dir, output_name, args.ext, args.dpi,
              **kwargs)


if __name__ == '__main__':
    main()
