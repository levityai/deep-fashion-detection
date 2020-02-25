import json
import tensorflow as tf
import io
from object_detection.utils import dataset_util
from PIL import Image


def get_classes():
    with open('train.json', 'r') as f:
        train = json.load(f)

    tags = []
    for img in train['images']:
        for box in img['boxes']:
            tags.append(box['tag'])

    tags = list(set(tags))
    return tags


def create_label_map():
    tags = get_classes()
    with open('training/data/label_map.pbtxt', 'w') as file:
        for i, tag in enumerate(tags):
            file.write('item\n')
            file.write('{\n')
            file.write('id: {}'.format(i + 1))
            file.write('\n')
            file.write("name: '{0}'".format(str(tag)))
            file.write('\n')
            file.write('}\n')


def create_tf_example(example, path):
    image = Image.open(path + example['path']).resize((512, 512))
    encoded_image_data = io.BytesIO()
    image.save(encoded_image_data, format='jpeg')
    encoded_image_data = encoded_image_data.getvalue()

    height = 512
    width = 512
    filename = example['path'].encode()
    image_format = 'jpeg'.encode()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    classes_text = []
    classes = []

    for box in example['boxes']:
        classes.append(int(box['tag']))
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
    with open('train.json', 'r') as f:
        train = json.load(f)

    with open('test.json', 'r') as f:
        test = json.load(f)

    with tf.io.TFRecordWriter('training/data/train.record') as writer:
        for example in train['images']:
            tf_example = create_tf_example(example, path='train/')
            writer.write(tf_example.SerializeToString())

    with tf.io.TFRecordWriter('training/data/val.record') as writer:
        for example in test['images']:
            tf_example = create_tf_example(example, path='test/')
            writer.write(tf_example.SerializeToString())


def get_num_classes():
    classes = get_classes()
    return len(classes)


def get_num_eval_examples():
    with open('test.json', 'r') as f:
        test = json.load(f)

    return len(test['images'])
