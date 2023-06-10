import os

bad_videos_file = 'kinetics_bad_videos.txt'
kinetics_root_dir = '../../dataset/Kinetics-400/videos/'


def main():
    f = open(bad_videos_file, 'r')
    videos = f.readlines()
    videos = [x.split('\n')[0] for x in videos]
    f.close()

    remove_count = 0
    for video in videos:
        path = os.path.join(kinetics_root_dir, video)
        if not os.path.exists(path):
            print(f'{path} not exist.')
            continue
        os.remove(path)
        remove_count += 1
    print(f'Removed {remove_count} bad videos.')


if __name__ == '__main__':
    main()
