import json
import tensorflow as tf
import io
from object_detection.utils import dataset_util
from PIL import Image
import os
import zipfile
from sklearn.model_selection import train_test_split

inputs_dir = os.getenv('VH_INPUTS_DIR')
outputs_dir = os.getenv('VH_OUTPUTS_DIR')


data_json = os.path.join(inputs_dir, 'data_json/data.json')
with open(data_json, 'r') as f:
    data = json.load(f)

train, test = train_test_split(data['images'], test_size=0.2)

tags = []
for img in train:
    for box in img['boxes']:
        tags.append(box['tag'])

tags = list(set(tags))


def create_label_map():
    with open('data/label_map.pbtxt', 'w') as file:
        for i, tag in enumerate(tags):
            file.write('item\n')
            file.write('{\n')
            file.write('id: {}'.format(i + 1))
            file.write('\n')
            file.write("name: '{0}'".format(str(tag)))
            file.write('\n')
            file.write('}\n')

    label_mapping = {
        'classes': dict(enumerate(tags)),
    }

    with open(os.path.join(outputs_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f)


def create_tf_example(example):
    image = Image.open('augmented/' + example['path'])
    width, height = image.size
    encoded_image_data = io.BytesIO()
    image.save(encoded_image_data, format='jpeg')
    encoded_image_data = encoded_image_data.getvalue()

    filename = example['path'].encode()
    image_format = 'jpeg'.encode()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for box in example['boxes']:
        classes.append(int(tags.index(box['tag'])))
        classes_text.append(str(box['tag']).encode())

        xmins.append(float(box['x_min']))
        xmaxs.append(float(box['x_max']))
        ymins.append(float(box['y_min']))
        ymaxs.append(float(box['y_max']))


    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_records():
    data_zip = os.path.join(inputs_dir, 'data/data.zip')
    zipfile.ZipFile(data_zip).extractall()

    with tf.python_io.TFRecordWriter('data/train.record') as writer:
        for example in train:
            if len(example['boxes']) > 0:
                tf_example = create_tf_example(example)
                writer.write(tf_example.SerializeToString())
            else:
                continue

    with tf.python_io.TFRecordWriter('data/val.record') as writer:
        for example in test:
            if len(example['boxes']) > 0:
                tf_example = create_tf_example(example)
                writer.write(tf_example.SerializeToString())
            else:
                continue


def get_num_classes():
    return len(tags)


def get_num_eval_examples():
    return len(test)


