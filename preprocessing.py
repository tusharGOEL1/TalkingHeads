import logging
import os
import pickle as pkl
import random
from multiprocessing import Pool

import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
from torch.utils.data import Dataset

K = 8


def preprocess_dataset(source, output, device='cpu', size=0, overwrite=False):
    logging.info('===== DATASET PRE-PROCESSING =====')
    logging.info(f'Running on {device.upper()}.')
    logging.info(f'Saving K+1 random frames from each video (K = {K}).')
    fa = FaceAlignment(LandmarksType._2D, device=device)

    video_list = get_video_list(source, size, output, overwrite=overwrite)

    logging.info(f'Processing {len(video_list)} videos...')

    init_pool(fa, output)
    counter = 1
    for v in video_list:
        process_video_folder(v)
        logging.info(f'{counter}/{len(video_list)}')
        counter += 1

    logging.info(f'All {len(video_list)} videos processed.')


def get_video_list(source, size, output, overwrite=True):
    already_processed = []
    if not overwrite:
        already_processed = [
            os.path.splitext(video_id)[0]
            for root, dirs, files in os.walk(output)
            for video_id in files
        ]

    video_list = []
    counter = 0
    for root, dirs, files in os.walk(source):
        if len(files) > 0 and os.path.basename(os.path.normpath(root)) not in already_processed:
            assert contains_only_videos(files) and len(dirs) == 0
            video_list.append((root, files))
            counter += 1
            if 0 < size <= counter:
                break

    return video_list


def init_pool(face_alignment, output):
    global _FA
    _FA = face_alignment
    global _OUT_DIR
    _OUT_DIR = output


def process_video_folder(video):
    folder, files = video

    try:
        assert contains_only_videos(files)
        frames = np.concatenate([extract_frames(os.path.join(folder, f)) for f in files])

        save_video(
            frames=select_random_frames(frames),
            video_id=os.path.basename(os.path.normpath(folder)),
            path=_OUT_DIR,
            face_alignment=_FA
        )
    except Exception as e:
        logging.error(f'Video {os.path.basename(os.path.normpath(folder))} could not be processed:\n{e}')


def contains_only_videos(files, extension='.mp4'):
    """
    Checks whether the files provided all end with the specified video extension.
    :param files: List of file names.
    :param extension: Extension that all files should have.
    :return: True if all files end with the given extension.
    """
    return len([x for x in files if os.path.splitext(x)[1] != extension]) == 0


def extract_frames(video):
    cap = cv2.VideoCapture(video)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = np.empty((n_frames, h, w, 3), np.dtype('uint8'))

    fn, ret = 0, True
    while fn < n_frames and ret:
        ret, img = cap.read()
        frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fn += 1

    cap.release()
    return frames


def select_random_frames(frames):
    S = random.sample(range(len(frames)), k=K+1)    
    return [frames[s] for s in S]


def save_video(path, video_id, frames, face_alignment):
    if not os.path.isdir(path):
        os.makedirs(path)

    data = []
    for i in range(len(frames)):
        x = frames[i]
        y = face_alignment.get_landmarks(x)[0]
        data.append({
            'frame': x,
            'landmarks': y,
        })

    filename = f'{video_id}.vid'
    pkl.dump(data, open(os.path.join(path, filename), 'wb'))
    logging.info(f'Saved file: {filename}')


if __name__ == '__main__':
    # preprocess_dataset("D:\Voxceleb2", "D:\VoxPickle", size=1000)
    pass