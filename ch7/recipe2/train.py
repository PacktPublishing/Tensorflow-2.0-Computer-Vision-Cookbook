import glob
import pathlib
import pickle

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.sequence import \
    pad_sequences

from ch7.recipe1.extract_features import ImageCaptionFeatureExtractor

BASE_PATH = (pathlib.Path.home() / '.keras' / 'datasets' /
             'flickr8k')
IMAGES_PATH = str(BASE_PATH / 'Images')
CAPTIONS_PATH = str(BASE_PATH / 'captions.txt')
OUTPUT_PATH = '.'


def load_paths_and_captions():
    image_paths = list(glob.glob(f'{IMAGES_PATH}/*.jpg'))

    with open(f'{CAPTIONS_PATH}', 'r') as f:
        text = f.read()
        lines = text.split('\n')

    mapping = {}
    for line in lines:
        if '.jpg' not in line:
            continue
        tokens = line.split(',', maxsplit=1)

        if len(line) < 2:
            continue

        image_id, image_caption = tokens
        image_id = image_id.split('.')[0]

        captions_per_image = mapping.get(image_id, [])
        captions_per_image.append(image_caption)

        mapping[image_id] = captions_per_image

    all_captions = []
    for image_path in image_paths:
        image_id = image_path.split('/')[-1].split('.')[0]
        all_captions.append(mapping[image_id][0])

    return image_paths, all_captions


def build_network(vocabulary_size,
                  max_sequence_length,
                  input_shape=(4096,)):
    feature_inputs = Input(shape=input_shape)
    x = Dropout(rate=0.5)(feature_inputs)
    x = Dense(units=256)(x)
    feature_output = ReLU()(x)

    sequence_inputs = Input(shape=(max_sequence_length,))
    y = Embedding(input_dim=vocabulary_size,
                  output_dim=256,
                  mask_zero=True)(sequence_inputs)
    y = Dropout(rate=0.5)(y)
    sequence_output = LSTM(units=256)(y)

    z = Add()([feature_output, sequence_output])
    z = Dense(units=256)(z)
    z = ReLU()(z)
    z = Dense(units=vocabulary_size)(z)
    outputs = Softmax()(z)

    return Model(inputs=[feature_inputs, sequence_inputs],
                 outputs=outputs)


def get_word_from_index(tokenizer, index):
    return tokenizer.index_word.get(index, None)


def produce_caption(model,
                    tokenizer,
                    image,
                    max_sequence_length):
    text = 'beginsequence'

    for i in range(max_sequence_length):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence],
                                 maxlen=max_sequence_length)

        prediction = model.predict([[image], sequence])
        index = np.argmax(prediction)

        word = get_word_from_index(tokenizer, index)

        if word is None:
            break

        text += f' {word}'

        if word == 'endsequence':
            break

    return text


def evaluate_model(model, features, captions, tokenizer,
                   max_seq_length):
    actual = []
    predicted = []

    for feature, caption in zip(features, captions):
        generated_caption = produce_caption(model,
                                            tokenizer,
                                            feature,
                                            max_seq_length)

        actual.append([caption.split(' ')])
        predicted.append(generated_caption.split(' '))

    for index, weights in enumerate([(1.0, 0, 0, 0),
                                     (0.5, 0.5, 0, 0),
                                     (0.3, 0.3, 0.3, 0),
                                     (0.25, 0.25, 0.25, 0.25)],
                                    start=1):
        b_score = corpus_bleu(actual, predicted, weights)
        print(f'BLEU-{index}: {b_score}')


image_paths, all_captions = load_paths_and_captions()

extractor_model = VGG16(weights='imagenet')
inputs = extractor_model.inputs
outputs = extractor_model.layers[-2].output
extractor_model = Model(inputs=inputs, outputs=outputs)

extractor = ImageCaptionFeatureExtractor(
    feature_extractor=extractor_model,
    output_path=OUTPUT_PATH)
extractor.extract_features(image_paths, all_captions)

pickled_data = []
for p in [f'{OUTPUT_PATH}/input_features.pickle',
          f'{OUTPUT_PATH}/input_sequences.pickle',
          f'{OUTPUT_PATH}/output_sequences.pickle']:
    with open(p, 'rb') as f:
        pickled_data.append(pickle.load(f))

input_feats, input_seqs, output_seqs = pickled_data

(train_input_feats, test_input_feats,
 train_input_seqs, test_input_seqs,
 train_output_seqs,
 test_output_seqs) = train_test_split(input_feats,
                                      input_seqs,
                                      output_seqs,
                                      train_size=0.8,
                                      random_state=9)

vocabulary_size = len(extractor.tokenizer.word_index) + 1
model = build_network(vocabulary_size,
                      extractor.max_seq_length)
model.compile(loss='categorical_crossentropy',
              optimizer='adam')

checkpoint_path = ('model-ep{epoch:03d}-loss{loss:.3f}-'
                   'val_loss{val_loss:.3f}.h5')
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')

checkpoints = sorted(list(glob.glob('./*.h5')), reverse=True)
if len(checkpoints) > 0:
    model = load_model(checkpoints[0])
else:
    EPOCHS = 30
    model.fit(x=[train_input_feats, train_input_seqs],
              y=train_output_seqs,
              epochs=EPOCHS,
              callbacks=[checkpoint],
              validation_data=([test_input_feats, test_input_seqs],
                               test_output_seqs))

with open(f'{OUTPUT_PATH}/data_mapping.pickle', 'rb') as f:
    data_mapping = pickle.load(f)

feats = [v['features'] for v in data_mapping.values()]
captions = [v['caption'] for v in data_mapping.values()]
evaluate_model(model,
               features=feats,
               captions=captions,
               tokenizer=extractor.tokenizer,
               max_seq_length=extractor.max_seq_length)
