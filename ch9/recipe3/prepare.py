# TODO Data: https://www.kaggle.com/mbkinaci/fruit-images-for-object-detection?
# TODO Install object detection API
# TODO Source: https://gilberttanner.com/blog/tensorflow-object-detection-with-tensorflow-2-creating-a-custom-model
# TODO Install Pandas
# TODO Install Pillow

import glob
import io
import os
from collections import namedtuple
from xml.etree import ElementTree as tree

import pandas as pd
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util


def encode_class(row_label):
    class_mapping = {'apple': 1, 'orange': 2, 'banana': 3}

    return class_mapping.get(row_label, None)


def split(df, group):
    Data = namedtuple('data', ['filename', 'object'])
    groups = df.groupby(group)
    return [Data(filename, groups.get_group(x))
            for filename, x
            in zip(groups.groups.keys(), groups.groups)]


def create_tf_example(group, path):
    groups_path = os.path.join(path, f'{group.filename}')
    with tf.gfile.GFile(groups_path, 'rb') as f:
        encoded_jpg = f.read()

    image = Image.open(io.BytesIO(encoded_jpg))
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(encode_class(row['class']))

    features = tf.train.Features(feature={
        'image/height':
            dataset_util.int64_feature(height),
        'image/width':
            dataset_util.int64_feature(width),
        'image/filename':
            dataset_util.bytes_feature(filename),
        'image/source_id':
            dataset_util.bytes_feature(filename),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymaxs),
        'image/object/class/text':
            dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':
            dataset_util.int64_list_feature(classes)
    })

    return tf.train.Example(features=features)


def bboxes_to_csv(path):
    xml_list = []

    bboxes_pattern = os.path.sep.join([path, '*.xml'])
    for xml_file in glob.glob(bboxes_pattern):
        t = tree.parse(xml_file)
        root = t.getroot()

        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)

    column_names = ['filename', 'width', 'height', 'class',
                    'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(xml_list, columns=column_names)
    return df


base = 'fruits'
for subset in ['test', 'train']:
    folder = os.path.sep.join([base, f'{subset}_zip', subset])

    labels_path = os.path.sep.join([base,
                                    f'{subset}_labels.csv'])
    bboxes_df = bboxes_to_csv(folder)
    bboxes_df.to_csv(labels_path, index=None)

    writer = (tf.python_io.
              TFRecordWriter(f'resources/{subset}.record'))
    examples = pd.read_csv(f'fruits/{subset}_labels.csv')
    grouped = split(examples, 'filename')

    path = os.path.join(f'fruits/{subset}_zip/{subset}')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()

# TODO Using this file: https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config
# TODO Downloaded model from here: http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
