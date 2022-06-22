import os
import numpy as np
import sys

import h5py

BASE_DIR = '/media/cesc/CESC/data/'
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import data_prep_util as data_prep_util
import indoor3d_util

# Constants
data_dir = os.path.join(ROOT_DIR, 'h5')
indoor3d_data_dir = '/media/cesc/CESC/data/'
NUM_POINT = 40960
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/anno_paths.txt')
data_label_files = [os.path.join(indoor3d_data_dir, line.rstrip()) for line in open(filelist)]
# output_dir = os.path.join(data_dir, 'indoor3d_sem_seg_hdf5_data')
output_dir = data_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):
    # global h5_batch_data, h5_batch_label
    batch_data_dim = [data.shape[0]] + data_dim
    batch_label_dim = [data.shape[0]] + label_dim
    h5_batch_data = np.zeros(batch_data_dim, dtype=np.float32)
    h5_batch_label = np.zeros(batch_label_dim, dtype=np.uint8)
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        data_prep_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return

class FError(Exception):
    pass


def save_batch_h5(fname, batch):
    try:
        fp = h5py.File(fname)
        coords = batch[:, :, 0:3]
        points = batch[:, :, 3:12]
        labels = batch[:, :, 12:14]
        fp.create_dataset('coords', data=coords, compression='gzip', dtype='float32')
        fp.create_dataset('points', data=points, compression='gzip', dtype='float32')
        fp.create_dataset('labels', data=labels, compression='gzip', dtype='int64')
        print('Stored {0} '.format(fname))
        fp.close()
        raise FError('error')
    except FError as e:
        print e


def insert(data, label, data_label_filename):
    base_name = os.path.basename(data_label_filename)
    name = os.path.splitext(base_name)[0]
    h5_filename = output_dir + '/' + name + '.h5'
    data_prep_util.save_h5(h5_filename, data, label, data_dtype, label_dtype)
    print('Stored {0} '.format(h5_filename))
    return

sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):
    path = os.path.normpath(data_label_filename)
    path1 = path.split(os.sep)[-3]
    name = path.split(os.sep)[-2]
    out_path = data_dir
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    h5_filename = out_path + '/' + path1 + '_' + name + '.h5'
    print ('save path : {0}'.format(h5_filename))
    file_exists = os.path.exists(h5_filename)
    if file_exists:
        print('{0} exists'.format(h5_filename))
        continue
    batch = indoor3d_util.room2blocks_wrapper_normalized_txt(data_label_filename, NUM_POINT, block_size=1.5, stride=0.8)
    print (batch.shape)
    fout_room.write(os.path.basename(out_path)[0:-4] + '\n')
    try:
        save_batch_h5(h5_filename, batch)
    except IOError:
        print "save error"


fout_room.close()
# for i, data_label_filename in enumerate(data_label_files):
#     file_name = os.listdir(data_label_filename)
#     for file in file_name:
#         txt = os.path.join(data_label_filename, file)
#         print (txt)
#         data, label = indoor3d_util.room2blocks_wrapper_normalized(txt, NUM_POINT, block_size=1.5, stride=0.8,
#                                                  random_sample=False, sample_num=None)
#         print('{0}, {1}'.format(data.shape, label.shape))
#         for _ in range(data.shape[0]):
#     fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')
#
#     sample_cnt += data.shape[0]
#     insert(data, label, data_label_filename)


