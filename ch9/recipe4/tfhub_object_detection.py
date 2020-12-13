import glob
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from object_detection.utils import visualization_utils as viz
from object_detection.utils.label_map_util import \
    create_category_index_from_labelmap


def load_image(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

    width, height = image.size
    shape = (1, height, width, 3)

    image = np.array(image.getdata())
    image = image.reshape(shape).astype('uint8')

    return image


def get_and_save_predictions(model, image_path):
    image = load_image(image_path)
    results = model(image)

    model_output = {k: v.numpy() for k, v in results.items()}

    boxes = model_output['detection_boxes'][0]
    classes = \
        model_output['detection_classes'][0].astype('int')
    scores = model_output['detection_scores'][0]

    clone = image.copy()
    viz.visualize_boxes_and_labels_on_image_array(
        image=clone[0],
        boxes=boxes,
        classes=classes,
        scores=scores,
        category_index=CATEGORY_IDX,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
        line_thickness=5
    )

    plt.figure(figsize=(24, 32))
    plt.imshow(clone[0])

    plt.savefig(f'output/{image_path.split("/")[-1]}')


labels_path = 'resources/mscoco_label_map.pbtxt'
CATEGORY_IDX = create_category_index_from_labelmap(labels_path)

MODEL_PATH = ('https://tfhub.dev/tensorflow/faster_rcnn/'
              'inception_resnet_v2_1024x1024/1')
model = hub.load(MODEL_PATH)

test_images_paths = glob.glob('test_images/*')
for image_path in test_images_paths:
    get_and_save_predictions(model, image_path)
