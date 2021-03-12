import h5py
import numpy as np
import argparse
from multiprocessing import Pool
from skvideo.io import vreader
from skimage.transform import resize
import os

# get home dir of dataset (furthest up file tree where you want to begin conversion)
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', '-d', help = 'parent path: all files in the \
        folder will be converted (recursive)')
args = parser.parse_args()

def convert_vid(path):
    out_path = path.split('.')[0] + '.hdf5'
    cap = vreader(path)
    with h5py.File(out_path, 'w') as f:
        needs_resize = False
        first_frame = next(cap)

        # NOTE this assumes original aspect ratio is 16:9
        if first_frame.shape[0] != 320:
            needs_resize = True
            first_frame = np.swapaxes(resize(first_frame, (320, 180)), 0, 1)

        f.create_dataset('vid_frames', data=np.expand_dims(first_frame, 0),
                maxshape=(None, first_frame.shape[0], first_frame.shape[1],
                first_frame.shape[2]), compression='gzip')
        for ind, frame in enumerate(cap):
            if needs_resize:
                frame = np.swapaxes(resize(frame, (320,180)), 0, 1)

            f['vid_frames'].resize((f['vid_frames'].shape[0] + 1), axis=0)
            f['vid_frames'][-1] = frame

    print('Wrote', out_path)

if __name__ == '__main__':
    # generate path list
    path_list = []
    for root, dirs, files in os.walk(args.data_dir):
        if len(dirs) == 0:
            for f in files:
                # only append mp4 and mov
                if f.endswith('mp4') or f.endswith('mov'):
                    path_list.append(os.path.join(root, f))
            continue

        # if we have multiple directories
        for d in dirs:
            for f in files:
                if f.endswith('mp4') or f.endswith('mov'):
                    path_list.append(os.path.join(root, f))

    with Pool(10) as p:
        p.map(convert_vid, path_list)
