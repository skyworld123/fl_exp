import argparse
import os
import pickle
from tqdm import tqdm


def load_data_from_pkl(path):
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def save_data_to_pkl(path, data):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def combine_pose_pkl(args):
    print('Collecting video paths...')
    class2label_path = 'kinetics400-class2label.pkl'
    required_splits = ['train', 'val']

    root = args.dir
    root_dirs = os.listdir(root)
    for split in required_splits:
        assert split in root_dirs
    # ensure class consistency
    class2label = load_data_from_pkl(class2label_path)
    classes = list(class2label.keys())
    for split in required_splits[1:]:
        split_classes = sorted(
            entry.name for entry in os.scandir(os.path.join(root, split)) if entry.is_dir())
        assert split_classes == classes, f'classes in {split} dir does not correspond with {required_splits[0]} dir'

    pkl_split = {}
    for split in required_splits:
        print(f'split {split}...')
        split_dir = os.path.join(root, split)
        video_list = []
        bar = tqdm(total=len(classes))
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            video_names = os.listdir(cls_dir)
            video_list.extend([f'{cls}/{name}' for name in video_names])
            bar.update()
        bar.close()
        pkl_split[split] = video_list

    print('Combining pickle files...')
    pkl_anno = []
    total_file_list = os.listdir(args.pkl_dir)
    for file in total_file_list:
        assert file.split('-')[-3] in required_splits

    for split in required_splits:
        print(f'split {split}...')
        file_list = [x for x in total_file_list if x.split('-')[-3] == split]
        assert len(file_list) > 0, f'File list for split {split} is empty'

        file_list = [x for x in file_list if x.endswith('.pkl')]
        cls_ranges = []
        for file in file_list:
            filename = os.path.splitext(file)[0]
            s, e = filename.split('-')[-2:]
            cls_ranges.append((int(s), int(e)))
        file_ranges = [(file_list[i], cls_ranges[i]) for i in range(len(file_list))]

        file_ranges.sort(key=lambda x: x[1][0])
        sorted_ranges = [x[1] for x in file_ranges]
        wrong_range_flag = False
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] != sorted_ranges[i + 1][0]:
                wrong_range_flag = True
        if sorted_ranges[-1][1] != len(class2label):
            wrong_range_flag = True
        if wrong_range_flag:
            raise ValueError(f'Wrong class ranges: {sorted_ranges}')
        file_list = [x[0] for x in file_ranges]

        bar = tqdm(total=len(file_list))
        for file in file_list:
            path = os.path.join(args.pkl_dir, file)
            pkl_anno_part = load_data_from_pkl(path)
            pkl_anno.extend(pkl_anno_part)
            bar.update()
        bar.close()

    pkl = {
        'split': pkl_split,
        'annotations': pkl_anno,
    }
    return pkl


def parse_args():
    parser = argparse.ArgumentParser(
        description='Combine multiple generated pose annotation pickle files (for kinetics).')
    parser.add_argument('--dir',
                        default='../../dataset/Kinetics-400/videos',
                        type=str, help='Root directory of Kinetics dataset.')
    parser.add_argument('--pkl_dir',
                        default='kinetics_poses',
                        type=str, help='root directory of pickle files')
    parser.add_argument('--output',
                        default='kinetics400.pkl',
                        type=str, help='path of the output pickle file')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    pkl = combine_pose_pkl(args)
    save_data_to_pkl(args.output, pkl)

    print('Finished.')


if __name__ == '__main__':
    # NUM_CLASSES = 400

    main()
