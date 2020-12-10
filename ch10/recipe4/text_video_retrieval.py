import math
import os
import uuid

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.keras.utils import get_file


def produce_embeddings(model, input_frames, input_words):
    frames = tf.cast(input_frames, dtype=tf.float32)
    frames = tf.constant(frames)
    video_model = model.signatures['video']
    video_embedding = video_model(frames)
    video_embedding = video_embedding['video_embedding']

    words = tf.constant(input_words)
    text_model = model.signatures['text']
    text_embedding = text_model(words)
    text_embedding = text_embedding['text_embedding']

    return video_embedding, text_embedding


def crop_center(frame):
    height, width = frame.shape[:2]
    smallest_dimension = min(width, height)

    x_start = (width // 2) - (smallest_dimension // 2)
    x_end = x_start + smallest_dimension

    y_start = (height // 2) - (smallest_dimension // 2)
    y_end = y_start + smallest_dimension

    roi = frame[y_start:y_end, x_start:x_end]
    return roi


def fetch_and_read_video(video_url,
                         max_frames=32,
                         resize=(224, 224)):
    extension = video_url.rsplit(os.path.sep,
                                 maxsplit=1)[-1]
    path = get_file(f'{str(uuid.uuid4())}.{extension}',
                    video_url,
                    cache_dir='.',
                    cache_subdir='.')
    capture = cv2.VideoCapture(path)

    frames = []
    while len(frames) <= max_frames:
        frame_read, frame = capture.read()

        if not frame_read:
            break

        frame = crop_center(frame)
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    capture.release()

    frames = np.array(frames)

    if len(frames) < max_frames:
        repetitions = math.ceil(
            float(max_frames) / len(frames))
        repetitions = int(repetitions)
        frames = frames.repeat(repetitions, axis=0)

    frames = frames[:max_frames]

    return frames / 255.0


URLS = [
    ('https://media.giphy.com/media/'
     'WWYSFIZo4fsLC/source.gif'),
    ('https://media.giphy.com/media/'
     'fwhIy2QQtu5vObfjrs/source.gif'),
    ('https://media.giphy.com/media/'
     'W307DdkjIsRHVWvoFE/source.gif'),
    ('https://media.giphy.com/media/'
     'FOcbaDiNEaqqY/source.gif'),
    ('https://media.giphy.com/media/'
     'VJwck53yG6y8s2H3Og/source.gif')]

VIDEOS = [fetch_and_read_video(url) for url in URLS]

QUERIES = ['beach', 'playing drums', 'airplane taking off',
           'biking', 'dog catching frisbee']

model = tfhub.load('https://tfhub.dev/deepmind/mil-nce/s3d/1')

video_emb, text_emb = produce_embeddings(model,
                                         np.stack(VIDEOS,
                                                  axis=0),
                                         np.array(QUERIES))

scores = np.dot(text_emb, tf.transpose(video_emb))

first_frames = [v[0] for v in VIDEOS]
first_frames = [cv2.cvtColor((f * 255.0).astype('uint8'),
                             cv2.COLOR_RGB2BGR) for f in
                first_frames]

for query, video, query_scores in zip(QUERIES, VIDEOS, scores):
    sorted_results = sorted(list(zip(QUERIES,
                                     first_frames,
                                     query_scores)),
                            key=lambda p: p[-1],
                            reverse=True)

    annotated_frames = []
    for i, (q, f, s) in enumerate(sorted_results, start=1):
        frame = f.copy()
        cv2.putText(frame,
                    f'#{i} - Score: {s:.2f}',
                    (8, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2)
        annotated_frames.append(frame)
    cv2.imshow(f'Results for query "{query}"',
               np.hstack(annotated_frames))
    cv2.waitKey(0)
