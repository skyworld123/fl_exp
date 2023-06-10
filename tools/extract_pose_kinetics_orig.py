import abc
import argparse
import cv2
import math
import numpy as np
import os
import pickle
from tqdm import tqdm

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

args = abc.abstractproperty()
args.det_config = 'mmdet_configs/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'
# args.det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_checkpoint = '../../pretrained/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'
args.det_score_thr = 0.5
args.pose_config = 'mmpose_configs/hrnet_w32_coco_256x192.py'
# args.pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501
args.pose_checkpoint = '../../pretrained/hrnet_w32_coco_256x192-c78dce93_20200708.pth'


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def save_data_to_pkl(path, data):
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


def extract_frame(video_path):
    vid = cv2.VideoCapture(video_path)
    frames = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        cnt += 1
        flag, frame = vid.read()
    return frames


def detection_inference(model, frames):
    results = []
    num_batches = math.ceil(len(frames) / BATCH_DET)

    for bi in range(num_batches):
        b_frames = frames[bi*BATCH_DET:(bi+1)*BATCH_DET]
        result = inference_detector(model, b_frames)
        result = [r[0][r[0][:, 4] >= args.det_score_thr] for r in result]
        results.extend(result)

    return results


def pose_inference(model, frames, det_results):
    num_frame = len(det_results)
    num_person = max([len(x) for x in det_results]) if num_frame > 0 else 0
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        # d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
    return kp


def kinetics_pose_extraction(args):
    """
    root:
        root/
        ├── split
        │   ├──  class1
        │   │   ├──  clip1.mp4
        │   │   ├──  clip2.mp4
        │   │   ├──  clip3.mp4
        │   │   ├──  ...
        │   ├──  class2
        │   │   ├──   clipx.mp4
        │   │   └── ...
    """
    root = args.dir

    print('Step 1/2: Collecting video paths...')
    required_splits = ['train', 'val']
    root_dirs = os.listdir(root)
    for split in required_splits:
        assert split in root_dirs
    # ensure class consistency
    classes = sorted(entry.name for entry in os.scandir(os.path.join(root, required_splits[0])) if entry.is_dir())
    assert len(classes) == NUM_CLASSES, \
        f'found {len(classes)} classes in {required_splits[0]} dir, but required {NUM_CLASSES}'
    for split in required_splits[1:]:
        split_classes = sorted(
            entry.name for entry in os.scandir(os.path.join(root, split)) if entry.is_dir())
        assert split_classes == classes, f'classes in {split} dir does not correspond with {required_splits[0]} dir'

    pkl_split = {}
    for split in required_splits:
        split_dir = os.path.join(root, split)
        video_list = []
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            video_names = os.listdir(cls_dir)
            video_list.extend([f'{cls}/{name}' for name in video_names])
        pkl_split[split] = video_list

    print('Step 2/2: Making annotations...')
    # load pretrained models
    det_model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert det_model.CLASSES[0] == 'person', ('We require you to use a detector '
                                              'trained on COCO')
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)

    pkl_anno = []
    for split in required_splits:
        print(f'Split {split}...')
        split_dir = os.path.join(root, split)

        for ci, cls in enumerate(classes):
            print(f'Class {cls} ({ci + 1}/{len(classes)})...')
            cls_dir = os.path.join(split_dir, cls)
            video_names = os.listdir(cls_dir)
            bar = tqdm(total=len(video_names))
            for name in video_names:
                video_path = os.path.join(cls_dir, name)

                frames = extract_frame(video_path)
                det_results = detection_inference(det_model, frames)
                pose_results = pose_inference(pose_model, frames, det_results)

                anno = dict()
                anno['keypoint'] = pose_results[..., :2]
                anno['keypoint_score'] = pose_results[..., 2]
                anno['frame_dir'] = f'{cls}/{name}'
                anno['total_frames'] = pose_results.shape[1]
                anno['original_shape'] = frames[0].shape[:2]
                anno['img_shape'] = frames[0].shape[:2]
                anno['label'] = ci
                pkl_anno.append(anno)

                bar.update()

    pkl = {
        'split': pkl_split,
        'annotations': pkl_anno,
    }
    return pkl


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for Kinetics dataset.')
    parser.add_argument('--dir',
                        default='../../temp/pose_extraction_test',
                        type=str, help='Root directory of Kinetics dataset.')
    parser.add_argument('--output_dir',
                        default='.',
                        type=str, help='Directory of the output pickle file.')
    parser.add_argument('--device',
                        default='cuda:0',
                        type=str, help='Device.')
    # parser.add_argument('--postproc', action='store_true')

    args = parser.parse_args()
    return args


def main():
    global_args = parse_args()
    args.device = global_args.device
    args.dir = global_args.dir
    args.output_dir = global_args.output_dir

    pkl = kinetics_pose_extraction(args)

    ensure_dir(args.output_dir)
    pkl_name = 'kinetics400.pkl'
    output_path = os.path.join(args.output_dir, pkl_name)
    print(f'Saving pose data to {output_path}...')
    save_data_to_pkl(output_path, pkl)
    print('Finished.')


if __name__ == '__main__':
    NUM_CLASSES = 400
    BATCH_DET = 2

    main()
