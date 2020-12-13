import glob
import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import ops
from object_detection.utils import visualization_utils as viz
from object_detection.utils.label_map_util import \
    create_category_index_from_labelmap


def load_image(path):
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

    width, height = image.size
    shape = (height, width, 3)

    image = np.array(image.getdata())
    image = image.reshape(shape).astype('uint8')

    return image


def infer_image(net, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model = net.signatures['serving_default']
    result = model(input_tensor)

    num_detections = int(result.pop('num_detections'))
    result = {key: value[0, :num_detections].numpy()
              for key, value in result.items()}
    result['num_detections'] = num_detections

    result['detection_classes'] = \
        result['detection_classes'].astype('int64')

    if 'detection_masks' in result:
        detection_masks_reframed = \
            ops.reframe_box_masks_to_image_masks(
                result['detection_masks'],
                result['detection_boxes'],
                image.shape[0],
                image.shape[1])

        detection_masks_reframed = \
            tf.cast(detection_masks_reframed > 0.5, tf.uint8)

        result['detection_masks_reframed'] = \
            detection_masks_reframed.numpy()

    return result


labels_path = 'resources/label_map.txt'
CATEGORY_IDX = \
    create_category_index_from_labelmap(labels_path,
                                        use_display_name=True)
model_path = 'resources/inference_graph/saved_model'
model = tf.saved_model.load(model_path)

test_images = list(glob.glob('fruits/test_zip/test/*.jpg'))
random.shuffle(test_images)
test_images = test_images[:3]

for image_path in test_images:
    image = load_image(image_path)
    result = infer_image(model, image)

    masks = result.get('detection_masks_reframed', None)
    viz.visualize_boxes_and_labels_on_image_array(
        image,
        result['detection_boxes'],
        result['detection_classes'],
        result['detection_scores'],
        CATEGORY_IDX,
        instance_masks=masks,
        use_normalized_coordinates=True,
        line_thickness=5)

    plt.figure(figsize=(24, 32))
    plt.imshow(image)
    plt.savefig(f'detections_{image_path.split("/")[-1]}')
