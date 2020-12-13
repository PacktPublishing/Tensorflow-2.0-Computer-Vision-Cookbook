# TODO pip install imutils
import cv2
import imutils
import numpy as np
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_resnet_v2 \
    import *
from tensorflow.keras.preprocessing.image import img_to_array


class ObjectDetector(object):
    def __init__(self, classifier,
                 preprocess_fn=lambda x: x,
                 input_size=(299, 299),
                 confidence=0.98,
                 window_step_size=16,
                 pyramid_scale=1.5,
                 roi_size=(200, 150),
                 nms_threshold=0.3):
        self.classifier = classifier
        self.preprocess_fn = preprocess_fn
        self.input_size = input_size
        self.confidence = confidence

        self.window_step_size = window_step_size

        self.pyramid_scale = pyramid_scale
        self.roi_size = roi_size
        self.nms_threshold = nms_threshold

    def sliding_window(self, image):
        for y in range(0,
                       image.shape[0],
                       self.window_step_size):
            for x in range(0,
                           image.shape[1],
                           self.window_step_size):
                y_slice = slice(y, y + self.roi_size[1], 1)
                x_slice = slice(x, x + self.roi_size[0], 1)

                yield x, y, image[y_slice, x_slice]

    def pyramid(self, image):
        yield image

        while True:
            width = int(image.shape[1] / self.pyramid_scale)
            image = imutils.resize(image, width=width)

            if (image.shape[0] < self.roi_size[1] or
                    image.shape[1] < self.roi_size[0]):
                break

            yield image

    def non_max_suppression(self, boxes, probabilities):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == 'i':
            boxes = boxes.astype(np.float)

        pick = []

        x_1 = boxes[:, 0]
        y_1 = boxes[:, 1]
        x_2 = boxes[:, 2]
        y_2 = boxes[:, 3]

        area = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        indexes = np.argsort(probabilities)

        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]
            pick.append(i)

            xx_1 = np.maximum(x_1[i], x_1[indexes[:last]])
            yy_1 = np.maximum(y_1[i], y_1[indexes[:last]])
            xx_2 = np.maximum(x_2[i], x_2[indexes[:last]])
            yy_2 = np.maximum(y_2[i], y_2[indexes[:last]])

            width = np.maximum(0, xx_2 - xx_1 + 1)
            height = np.maximum(0, yy_2 - yy_1 + 1)

            overlap = (width * height) / area[indexes[:last]]

            redundant_boxes = \
                np.where(overlap > self.nms_threshold)[0]
            to_delete = np.concatenate(
                ([last], redundant_boxes))
            indexes = np.delete(indexes, to_delete)

        return boxes[pick].astype(np.int)

    def detect(self, image):
        rois = []
        locations = []

        for img in self.pyramid(image):
            scale = image.shape[1] / float(img.shape[1])

            for x, y, roi_original in \
                    self.sliding_window(img):
                x = int(x * scale)
                y = int(y * scale)
                w = int(self.roi_size[0] * scale)
                h = int(self.roi_size[1] * scale)

                roi = cv2.resize(roi_original,
                                 self.input_size)
                roi = img_to_array(roi)
                roi = self.preprocess_fn(roi)

                rois.append(roi)
                locations.append((x, y, x + w, y + h))

        rois = np.array(rois, dtype=np.float32)

        predictions = self.classifier.predict(rois)
        predictions = \
            imagenet_utils.decode_predictions(predictions,
                                              top=1)

        labels = {}
        for i, pred in enumerate(predictions):
            _, label, proba = pred[0]

            if proba >= self.confidence:
                box = locations[i]

                label_detections = labels.get(label, [])
                label_detections.append({'box': box,
                                         'proba': proba})
                labels[label] = label_detections

        return labels


model = InceptionResNetV2(weights='imagenet',
                          include_top=True)
object_detector = ObjectDetector(model, preprocess_input)

image = cv2.imread('dog.jpg')
image = imutils.resize(image, width=600)
labels = object_detector.detect(image)

GREEN = (0, 255, 0)
for i, label in enumerate(labels.keys()):
    clone = image.copy()

    for detection in labels[label]:
        box = detection['box']
        probability = detection['proba']

        x_start, y_start, x_end, y_end = box
        cv2.rectangle(clone, (x_start, y_start),
                      (x_end, y_end), (0, 255, 0), 2)

    cv2.imwrite(f'Before_{i}.jpg', clone)

    clone = image.copy()
    boxes = np.array([d['box'] for d in labels[label]])
    probas = np.array([d['proba'] for d in labels[label]])
    boxes = object_detector.non_max_suppression(boxes,
                                                probas)

    for x_start, y_start, x_end, y_end in boxes:
        cv2.rectangle(clone,
                      (x_start, y_start),
                      (x_end, y_end),
                      GREEN,
                      2)

        if y_start - 10 > 10:
            y = y_start - 10
        else:
            y = y_start + 10

        cv2.putText(clone,
                    label,
                    (x_start, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .45,
                    GREEN,
                    2)

    cv2.imwrite(f'After_{i}.jpg', clone)
