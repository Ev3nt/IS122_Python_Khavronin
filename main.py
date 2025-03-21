import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers, models

image_size = [image_width, image_height] = (224, 224)

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return boxes, labels

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def prepare_dataset(image_paths, xml_paths):
    images = []
    bboxes = []
    labels = []

    for img_path, xml_path in zip(image_paths, xml_paths):
        image = load_image(img_path)
        boxes, lbls = parse_xml(xml_path)

        images.append(image)
        bboxes.append(boxes)
        labels.append(lbls)

    bboxes = tf.ragged.constant(bboxes, dtype=tf.float32)
    labels = tf.ragged.constant(labels, dtype=tf.string)
    
    return tf.data.Dataset.from_tensor_slices((images, bboxes, labels))

def preprocess_data(image, bboxes, labels):
    image = tf.image.resize(image, image_size)
    image = image / 255.0

    return image, bboxes, labels

def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)

    bbox_output = layers.Dense(4, activation='linear', name='bbox_output')(x)
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)

    model = models.Model(inputs=inputs, outputs=[bbox_output, class_output])

    return model

files = os.listdir('dataset')
xml_paths = ['dataset/' + x for x in list(filter(lambda x: (os.path.splitext(x)[1] == '.xml'), files))]
image_paths = ['dataset/' + x for x in list(filter(lambda x: (os.path.splitext(x)[1] == '.png'), files))]

dataset = prepare_dataset(image_paths, xml_paths)
dataset = dataset.map(preprocess_data).batch(32)

input_shape = (image_width, image_height, 3)
num_classes = 10
model = build_model(input_shape, num_classes)
model.summary()

model.compile(
    optimizer='adam',
    loss={
        'bbox_output': 'mse',
        'class_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'class_output': 'accuracy'
    }
)

history = model.fit(dataset, epochs=10)