import tensorflow as tf
import numpy as np
import glob
# use following commands when 'Segmentation fault' error occurs
# import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt
from PIL import Image
import random


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw

def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """
    feature = {'label': _float_feature(lab),
               'encoded': _bytes_feature(img)}

    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example.SerializeToString()


def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """

    # datadir as filename.

    filenames = glob.glob(imagedir)
    writer = tf.io.TFRecordWriter(datadir)
    random.shuffle(filenames)

    for i in range(len(filenames)):
        lab = int(filenames[i][-11])
        filename = filenames[i]

        img_data = _image_as_bytes(filename)

        example = make_example(img_data, lab)
        writer.write(example)
    writer.close()




def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """

    #filename = glob.glob(folder)
    filename = glob.glob(folder + '/*')

    filename_queue = tf.train.string_input_producer(filename, num_epochs=epoch)

    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)

    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, ''),
                      'label': tf.FixedLenFeature([], tf.float32, 0)}

    features = tf.parse_single_example(serialized_example, features=key_to_feature)


    img = tf.decode_raw(features['encoded'], tf.uint8)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [28,28,1])

    lab = tf.cast(features['label'], dtype=tf.int32)

    min_after_dequeue = 10

    img, lab = tf.train.shuffle_batch([img, lab], batch_size=batch,
                                      capacity = min_after_dequeue + 3*batch,
                                      num_threads =1,
                                      min_after_dequeue=min_after_dequeue)
    lab_ = tf.one_hot(lab, depth=10)
    lab_ = tf.reshape(lab_, shape=[-1, 10])

    return img, lab_
