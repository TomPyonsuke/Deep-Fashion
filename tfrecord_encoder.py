from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import izip
import math
import os
import random
import sys
import tensorflow as tf
import numpy as np
import dataset_utils
import cv2

#Dataset Root Directory Path
DATASET_DIR = '/media/tjiang/Elements/ImageClassification/Deep_Fashion/'

# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_labels(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  list_category_img = os.path.join(dataset_dir, 'Anno/list_category_img.txt')
  list_eval_partition = os.path.join(dataset_dir, 'Eval/list_eval_partition.txt')

  training_filenames = []
  validation_filenames = []
  test_filenames = []
  training_labels = []
  validation_labels = []
  test_labels = []
  with open(list_category_img) as f1, open(list_eval_partition) as f2:
    f1_line_cnt = int(f1.readline().strip())
    f2_line_cnt = int(f2.readline().strip())
    f1.next()
    f2.next()
    cnt = 0; train_cnt = 0; val_cnt = 0; test_cnt = 0;
    for x, y in izip(f1, f2):
      photo_filename1, label = x.strip().split()
      photo_filename2, partition = y.strip().split()
      assert photo_filename1 == photo_filename2
      photo_filename1 = os.path.join(dataset_dir, photo_filename1)
      if partition == 'train':
        training_filenames.append(photo_filename1)
        training_labels.append(int(label))
        train_cnt += 1
      elif partition == 'val':
        validation_filenames.append(photo_filename1)
        validation_labels.append(int(label))
        val_cnt += 1
      else:
        test_filenames.append(photo_filename1)
        test_labels.append(int(label))
        test_cnt += 1
      cnt += 1
    assert cnt == f1_line_cnt
    assert cnt == f2_line_cnt

    print ("Number of training images: %d\n \
            Number of validation images: %d\n \
            Number of test images: %d\n" % (train_cnt, val_cnt, test_cnt))

  return training_filenames, training_labels, validation_filenames, \
         validation_labels, test_filenames, test_labels

def _shuffle_dataset(data, labels):
    assert len(data) ==  len(labels)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    data = np.array(data)
    shuffled_data = data[indices]
    labels = np.array(labels)
    shuffled_labels = labels[indices]
    return shuffled_data, shuffled_labels

def _get_categories_and_types(dataset_dir):
    filepath = os.path.join(dataset_dir, 'Anno/list_category_cloth.txt')
    categories = {}
    with open(filepath) as f:
        category_cnt = int(f.readline().strip())
        f.next()
        cnt = 0
        for line in f:
            category, type = line.strip().split()
            categories[category] = type
            cnt += 1
        assert cnt == category_cnt
    return categories

def _get_bbox(dataset_dir):
    filepath = os.path.join(dataset_dir, 'Anno/list_bbox.txt')
    bbox_dict = {}
    with open(filepath) as f:
        bbox_cnt = int(f.readline().strip())
        f.next()
        cnt = 0
        for line in f:
            photo_filename, x_min, y_min, x_max, y_max = line.strip().split()
            photo_filename = os.path.join(dataset_dir, photo_filename)
            bbox_dict[photo_filename] = [int(x_min), int(y_min), int(x_max), int(y_max)]
            cnt += 1
        assert cnt == bbox_cnt
    return bbox_dict

def _get_dataset_filename(dataset_dir, split_name):
  output_filename = '%s-deep-fashion.tfrecord' % (split_name)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, labels, bbox_dict, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation', 'test']

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:
        output_filename = _get_dataset_filename(dataset_dir, split_name)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          for i in range(len(filenames)):
            sys.stdout.write('\r>> Converting image %d/%d' % (
                i+1, len(filenames)))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)
            x_min = bbox_dict[filenames[i]][0] / width
            y_min = bbox_dict[filenames[i]][1] / height
            x_max = bbox_dict[filenames[i]][2] / width
            y_max = bbox_dict[filenames[i]][3] / height
            example = dataset_utils.image_to_tfexample(
                image_data, 'jpeg', height, width, labels[i], [[x_min], [y_min], [x_max], [y_max]])
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation', 'test']:
    output_filename = _get_dataset_filename(dataset_dir, split_name)
    if not tf.gfile.Exists(output_filename):
       return False
  return True

def run(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    #return

  train_files, train_labels, val_files, \
  val_labels, test_files, test_labels = _get_filenames_and_labels(dataset_dir)

  category_list = _get_categories_and_types(dataset_dir)
  bbox_dict = _get_bbox(dataset_dir)
  # Shuffle records
  train_files, train_labels = _shuffle_dataset(train_files, train_labels)
  val_files, val_labels = _shuffle_dataset(val_files, val_labels)
  test_files, test_labels = _shuffle_dataset(test_files, test_labels)

  # Convert training, validation and test sets.
  _convert_dataset('train', train_files, train_labels,
                   bbox_dict, dataset_dir)
  _convert_dataset('validation', val_files, val_labels,
                   bbox_dict, dataset_dir)
  _convert_dataset('test', test_files, test_labels,
                   bbox_dict, dataset_dir)

  # Finally, write the labels file:
  #labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  #dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the deep fashion dataset!')

if __name__ == '__main__':
  run(DATASET_DIR)