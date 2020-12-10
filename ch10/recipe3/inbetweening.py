import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow_datasets.core import SplitGenerator
from tensorflow_datasets.video.bair_robot_pushing import \
    BairRobotPushingSmall


def plot_first_and_last_for_sample(frames, batch_size):
    for i in range(4):
        plt.subplot(batch_size, 2, 1 + 2 * i)
        plt.imshow(frames[i, 0] / 255.)
        plt.title(f'Video {i}: first frame')
        plt.axis('off')

        plt.subplot(batch_size, 2, 2 + 2 * i)
        plt.imshow(frames[i, 1] / 255.)
        plt.title(f'Video {i}: last frame')
        plt.axis('off')


def plot_generated_frames_for_sample(gen_videos):
    for video_id in range(4):
        fig = plt.figure(figsize=(10 * 2, 2))
        for frame_id in range(1, 16):
            ax = fig.add_axes(
                [frame_id / 16., 0, (frame_id + 1) / 16., 1],
                xmargin=0, ymargin=0)
            ax.imshow(gen_videos[video_id, frame_id])
            ax.axis('off')


def split_gen_func(data_path):
    return [SplitGenerator(name='test',
                           gen_kwargs={'filedir': data_path})]


DATA_PATH = str(pathlib.Path.home() / '.keras' / 'datasets' /
                'bair_robot_pushing')

builder = BairRobotPushingSmall()
builder._split_generators = lambda _: split_gen_func(DATA_PATH)
builder.download_and_prepare()

BATCH_SIZE = 16

dataset = builder.as_dataset(split='test')
test_videos = dataset.batch(BATCH_SIZE)

for video in test_videos:
    first_batch = video
    break

input_frames = first_batch['image_aux1'][:, ::15]
input_frames = tf.cast(input_frames, tf.float32)

model_path = 'https://tfhub.dev/google/tweening_conv3d_bair/1'
model = tfhub.load(model_path)
model = model.signatures['default']

middle_frames = model(input_frames)['default']
middle_frames = middle_frames / 255.0

generated_videos = np.concatenate(
    [input_frames[:, :1] / 255.0,  # All first frames
     middle_frames,  # All inbetween frames
     input_frames[:, 1:] / 255.0],  # All last frames
    axis=1)

plt.figure(figsize=(4, 2 * BATCH_SIZE))
plot_first_and_last_for_sample(input_frames, BATCH_SIZE)
plot_generated_frames_for_sample(generated_videos)
plt.show()
