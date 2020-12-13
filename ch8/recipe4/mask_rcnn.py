import glob
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from object_detection.utils import ops
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

    detection_masks = model_output['detection_masks'][0]
    detection_masks = tf.convert_to_tensor(detection_masks)

    detection_boxes = model_output['detection_boxes'][0]
    detection_boxes = tf.convert_to_tensor(detection_boxes)

    detection_masks_reframed = \
        ops.reframe_box_masks_to_image_masks(detection_masks,
                                             detection_boxes,
                                             image.shape[1],
                                             image.shape[2])
    detection_masks_reframed = \
        tf.cast(detection_masks_reframed > 0.5, tf.uint8)

    model_output['detection_masks_reframed'] = \
        detection_masks_reframed.numpy()

    boxes = model_output['detection_boxes'][0]
    classes = \
        model_output['detection_classes'][0].astype('int')
    scores = model_output['detection_scores'][0]
    masks = model_output['detection_masks_reframed']

    image_with_mask = image.copy()
    viz.visualize_boxes_and_labels_on_image_array(
        image=image_with_mask[0],
        boxes=boxes,
        classes=classes,
        scores=scores,
        category_index=CATEGORY_IDX,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.30,
        agnostic_mode=False,
        instance_masks=masks,
        line_thickness=5
    )

    plt.figure(figsize=(24, 32))
    plt.imshow(image_with_mask[0])

    plt.savefig(f'output/{image_path.split("/")[-1]}')


labels_path = 'resources/mscoco_label_map.pbtxt'
CATEGORY_IDX = create_category_index_from_labelmap(labels_path)

MODEL_PATH = ('https://tfhub.dev/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1')
mask_rcnn = hub.load(MODEL_PATH)

test_images_paths = glob.glob('test_images/*')
for image_path in test_images_paths:
    get_and_save_predictions(mask_rcnn, image_path)
