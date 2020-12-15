import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.sequence import \
    pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from ch7.recipe1.extract_features import ImageCaptionFeatureExtractor


def get_word_from_index(tokenizer, index):
    return tokenizer.index_word.get(index, None)


def produce_caption(model,
                    tokenizer,
                    image,
                    max_sequence_length):
    text = 'beginsequence'

    for _ in range(max_sequence_length):
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


extractor_model = VGG16(weights='imagenet')
inputs = extractor_model.inputs
outputs = extractor_model.layers[-2].output
extractor_model = Model(inputs=inputs, outputs=outputs)

extractor = ImageCaptionFeatureExtractor(
    feature_extractor=extractor_model)

with open('data_mapping.pickle', 'rb') as f:
    data_mapping = pickle.load(f)

captions = [v['caption'] for v in data_mapping.values()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
max_seq_length = extractor._get_max_seq_length(captions)

model = load_model('model-ep003-loss3.847-val_loss4.328.h5')

for idx, image_path in enumerate(glob.glob('*.jpg'), start=1):
    img_feats = (extractor
                 .extract_image_features(image_path))

    description = produce_caption(model,
                                  tokenizer,
                                  img_feats,
                                  max_seq_length)
    description = (description
                   .replace('beginsequence', '')
                   .replace('endsequence', ''))

    image = plt.imread(image_path)

    plt.imshow(image)
    plt.title(description)
    plt.savefig(f'{idx}.jpg')
