import glob
import os
import pathlib
import pickle
from string import punctuation

import numpy as np
import tqdm
from tensorflow.keras.applications.vgg16 import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.preprocessing.sequence import \
    pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm


class ImageCaptionFeatureExtractor(object):
    def __init__(self,
                 output_path='.',
                 start_token='beginsequence',
                 end_token='endsequence',
                 feature_extractor=None,
                 input_shape=(224, 224, 3)):
        self.input_shape = input_shape

        if feature_extractor is None:
            input = Input(shape=input_shape)
            self.feature_extractor = VGG16(input_tensor=input,
                                           weights='imagenet',
                                           include_top=False)
        else:
            self.feature_extractor = feature_extractor

        self.output_path = output_path
        self.start_token = start_token
        self.end_token = end_token
        self.tokenizer = Tokenizer()
        self.max_seq_length = None

    def extract_image_features(self, image_path):
        image = load_img(image_path,
                         target_size=self.input_shape[:2])
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        return self.feature_extractor.predict(image)[0]

    def _clean_captions(self, captions):
        def remove_punctuation(word):
            translation = str.maketrans('', '',
                                        punctuation)
            return word.translate(translation)

        def is_valid_word(word):
            return len(word) > 1 and word.isalpha()

        cleaned_captions = []
        for caption in captions:
            caption = caption.lower().split(' ')
            caption = map(remove_punctuation, caption)
            caption = filter(is_valid_word, caption)

            cleaned_caption = f'{self.start_token} ' \
                              f'{" ".join(caption)} ' \
                              f'{self.end_token}'
            cleaned_captions.append(cleaned_caption)

        return cleaned_captions

    def _get_max_seq_length(self, captions):
        max_sequence_length = -1

        for caption in captions:
            caption_length = len(caption.split(' '))
            max_sequence_length = max(max_sequence_length,
                                      caption_length)

        return max_sequence_length

    def extract_features(self, images_path, captions):
        assert len(images_path) == len(captions)

        captions = self._clean_captions(captions)
        self.max_seq_length = self._get_max_seq_length(captions)
        self.tokenizer.fit_on_texts(captions)

        data_mapping = {}
        print('\nExtracting features...')
        for i in tqdm(range(len(images_path))):
            image_path = images_path[i]
            caption = captions[i]

            feats = self.extract_image_features(image_path)

            image_id = image_path.split(os.path.sep)[-1]
            image_id = image_id.split('.')[0]

            data_mapping[image_id] = {
                'features': feats,
                'caption': caption
            }

        out_path = f'{self.output_path}/data_mapping.pickle'
        with open(out_path, 'wb') as f:
            pickle.dump(data_mapping, f, protocol=4)

        self._create_sequences(data_mapping)

    def _create_sequences(self, mapping):
        num_classes = len(self.tokenizer.word_index) + 1

        in_feats = []
        in_seqs = []
        out_seqs = []

        print('\nCreating sequences...')
        for _, data in tqdm(mapping.items()):
            feature = data['features']
            caption = data['caption']

            seq, = self.tokenizer.texts_to_sequences([caption])

            for i in range(1, len(seq)):
                input_seq = seq[:i]
                input_seq, = pad_sequences([input_seq],
                                           self.max_seq_length)

                # ONE HOT ENCODING
                out_seq = seq[i]
                out_seq = to_categorical([out_seq],
                                         num_classes)[0]

                in_feats.append(feature)
                in_seqs.append(input_seq)
                out_seqs.append(out_seq)

        file_paths = [
            f'{self.output_path}/input_features.pickle',
            f'{self.output_path}/input_sequences.pickle',
            f'{self.output_path}/output_sequences.pickle']
        sequences = [in_feats,
                     in_seqs,
                     out_seqs]

        for path, seq in zip(file_paths, sequences):
            with open(path, 'wb') as f:
                pickle.dump(np.array(seq), f, protocol=4)


BASE_PATH = (pathlib.Path.home() / '.keras' / 'datasets' /
             'flickr8k')
IMAGES_PATH = str(BASE_PATH / 'Images')
CAPTIONS_PATH = str(BASE_PATH / 'captions.txt')

extractor = ImageCaptionFeatureExtractor(output_path='.')

image_paths = list(glob.glob(f'{IMAGES_PATH}/*.jpg'))

with open(CAPTIONS_PATH, 'r') as f:
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
    image_id = image_id.split('.')[0]  # Remove filename from image id

    captions_per_image = mapping.get(image_id, [])
    captions_per_image.append(image_caption)

    mapping[image_id] = captions_per_image

captions = []
for image_path in image_paths:
    image_id = image_path.split('/')[-1].split('.')[0]

    captions.append(mapping[image_id][0])

extractor.extract_features(image_paths, captions)
