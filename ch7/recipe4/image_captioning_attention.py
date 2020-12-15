import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.applications.inception_v3 import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import \
    SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import \
    pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import get_file

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (299, 299))
    image = preprocess_input(image)

    return image, image_path


def get_max_length(tensor):
    return max(len(t) for t in tensor)


def load_image_and_caption(image_name, caption):
    image_name = image_name.decode('utf-8').split('/')[-1]
    image_tensor = np.load(f'./{image_name}.npy')
    return image_tensor, caption


class BahdanauAttention(Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        score = tf.nn.tanh(self.W1(features) +
                           self.W2(hidden_with_time_axis))

        attention_w = tf.nn.softmax(self.V(score), axis=1)

        ctx_vector = attention_w * features
        ctx_vector = tf.reduce_sum(ctx_vector, axis=1)

        return ctx_vector, attention_w


class CNNEncoder(Model):
    def __init__(self, embedding_dim):
        super(CNNEncoder, self).__init__()
        self.fc = Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)

        return x


class RNNDecoder(Model):
    def __init__(self, embedding_size, units, vocab_size):
        super(RNNDecoder, self).__init__()
        self.units = units

        self.embedding = Embedding(vocab_size, embedding_size)
        self.gru = GRU(self.units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')
        self.fc1 = Dense(self.units)
        self.fc2 = Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = \
            self.attention(features, hidden)

        x = self.embedding(x)

        expanded_context = tf.expand_dims(context_vector, 1)
        x = Concatenate(axis=-1)([expanded_context, x])

        output, state = self.gru(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


class ImageCaptioner(object):
    def __init__(self, embedding_size, units, vocab_size,
                 tokenizer):
        self.tokenizer = tokenizer
        self.encoder = CNNEncoder(embedding_size)
        self.decoder = RNNDecoder(embedding_size, units,
                                  vocab_size)

        self.optimizer = Adam()
        self.loss = SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none')

    def loss_function(self, real, predicted):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        _loss = self.loss(real, predicted)

        mask = tf.cast(mask, dtype=_loss.dtype)
        _loss *= mask

        return tf.reduce_mean(_loss)

    @tf.function
    def train_step(self, image_tensor, target):
        loss = 0

        hidden = self.decoder.reset_state(target.shape[0])
        start_token_idx = self.tokenizer.word_index['<start>']
        init_batch = [start_token_idx] * target.shape[0]
        decoder_input = tf.expand_dims(init_batch, 1)

        with tf.GradientTape() as tape:
            features = self.encoder(image_tensor)

            for i in range(1, target.shape[1]):
                preds, hidden, _ = self.decoder(decoder_input,
                                                features,
                                                hidden)
                loss += self.loss_function(target[:, i],
                                           preds)
                decoder_input = tf.expand_dims(target[:, i], 1)

        total_loss = loss / int(target.shape[1])

        trainable_vars = (self.encoder.trainable_variables +
                          self.decoder.trainable_variables)
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients,
                                           trainable_vars))

        return loss, total_loss

    def train(self, dataset, epochs, num_steps):
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0

            for batch, (image_tensor, target) \
                    in enumerate(dataset):
                batch_loss, step_loss = \
                    self.train_step(image_tensor, target)
                total_loss += step_loss

                if batch % 100 == 0:
                    loss = batch_loss.numpy()
                    loss = loss / int(target.shape[1])
                    print(f'Epoch {epoch + 1}, batch {batch},'
                          f' loss {loss:.4f}')

            print(f'Epoch {epoch + 1},'
                  f' loss {total_loss / num_steps:.6f}')
            epoch_time = time.time() - start
            print(f'Time taken: {epoch_time} seconds. \n')


# Download caption annotation files
INPUT_DIR = os.path.abspath('.')
annots_folder = '/annotations/'
if not os.path.exists(INPUT_DIR + annots_folder):
    origin_url = ('http://images.cocodataset.org/annotations'
                  '/annotations_trainval2014.zip')
    cache_subdir = os.path.abspath('.')
    annots_zip = get_file('all_captions.zip',
                          cache_subdir=cache_subdir,
                          origin=origin_url,
                          extract=True)
    annots_file = (os.path.dirname(annots_zip) +
                   '/annotations/captions_train2014.json')
    os.remove(annots_zip)
else:
    annots_file = (INPUT_DIR +
                   '/annotations/captions_train2014.json')

# Download image files
image_folder = '/train2014/'
if not os.path.exists(INPUT_DIR + image_folder):
    origin_url = ('http://images.cocodataset.org/zips/'
                  'train2014.zip')
    cache_subdir = os.path.abspath('.')
    image_zip = get_file('train2014.zip',
                         cache_subdir=cache_subdir,
                         origin=origin_url,
                         extract=True)
    PATH = os.path.dirname(image_zip) + image_folder
    os.remove(image_zip)
else:
    PATH = INPUT_DIR + image_folder

# Read the JSON file
with open(annots_file, 'r') as f:
    annotations = json.load(f)

# Store all_captions and image names in vectors
captions = []
image_paths = []

for annotation in annotations['annotations']:
    caption = '<start>' + annotation['caption'] + ' <end>'
    image_id = annotation['image_id']
    image_path = f'{PATH}COCO_train2014_{image_id:012d}.jpg'

    image_paths.append(image_path)
    captions.append(caption)

train_captions, train_img_paths = shuffle(captions,
                                          image_paths,
                                          random_state=42)

SAMPLE_SIZE = 30000
train_captions = train_captions[:SAMPLE_SIZE]
train_img_paths = train_img_paths[:SAMPLE_SIZE]

train_images = sorted(set(train_img_paths))

feature_extractor = InceptionV3(include_top=False,
                                weights='imagenet')
feature_extractor = Model(feature_extractor.input,
                          feature_extractor.layers[-1].output)

BATCH_SIZE = 8
image_dataset = (tf.data.Dataset
                 .from_tensor_slices(train_images)
                 .map(load_image, num_parallel_calls=AUTOTUNE)
                 .batch(BATCH_SIZE))

for image, path in image_dataset:
    batch_features = feature_extractor.predict(image)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0],
                                 -1,
                                 batch_features.shape[3]))

    for batch_feature, p in zip(batch_features, path):
        feature_path = p.numpy().decode('UTF-8')
        image_name = feature_path.split('/')[-1]
        np.save(f'./{image_name}', batch_feature.numpy())

top_k = 5000
filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
tokenizer = Tokenizer(num_words=top_k,
                      oov_token='<unk>',
                      filters=filters)
tokenizer.fit_on_texts(train_captions)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

train_seqs = tokenizer.texts_to_sequences(train_captions)
captions_seqs = pad_sequences(train_seqs, padding='post')

max_length = get_max_length(train_seqs)

(images_train, images_val, caption_train, caption_val) = \
    train_test_split(train_img_paths,
                     captions_seqs,
                     test_size=0.2,
                     random_state=42)

BATCH_SIZE = 64
BUFFER_SIZE = 1000
dataset = (tf.data.Dataset
           .from_tensor_slices((images_train, caption_train))
           .map(lambda i1, i2:
                tf.numpy_function(
                    load_image_and_caption,
                    [i1, i2],
                    [tf.float32, tf.int32]),
                num_parallel_calls=AUTOTUNE)
           .shuffle(BUFFER_SIZE)
           .batch(BATCH_SIZE)
           .prefetch(buffer_size=AUTOTUNE))

image_captioner = ImageCaptioner(embedding_size=256,
                                 units=512,
                                 vocab_size=top_k + 1,
                                 tokenizer=tokenizer)

EPOCHS = 30
num_steps = len(images_train) // BATCH_SIZE
image_captioner.train(dataset, EPOCHS, num_steps)


def evaluate(encoder, decoder, tokenizer, image, max_length,
             attention_shape):
    attention_plot = np.zeros((max_length,
                               attention_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    image_tensor_val = feature_extractor(temp_input)
    image_tensor_val = tf.reshape(image_tensor_val,
                                  (image_tensor_val.shape[0],
                                   -1,
                                   image_tensor_val.shape[3]))

    feats = encoder(image_tensor_val)

    start_token_idx = tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token_idx], 0)
    result = []

    for i in range(max_length):
        (preds, hidden, attention_w) = \
            decoder(dec_input, feats, hidden)

        attention_plot[i] = tf.reshape(attention_w,
                                       (-1,)).numpy()

        pred_id = tf.random.categorical(preds,
                                        1)[0][0].numpy()
        result.append(tokenizer.index_word[pred_id])

        if tokenizer.index_word[pred_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([pred_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result,
                   attention_plot, output_path):
    tmp_image = np.array(load_image(image)[0])

    fig = plt.figure(figsize=(10, 10))

    for l in range(len(result)):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len(result) // 2,
                             len(result) // 2,
                             l + 1)
        ax.set_title(result[l])
        image = ax.imshow(tmp_image)

        ax.imshow(temp_att,
                  cmap='gray',
                  alpha=0.6,
                  extent=image.get_extent())

    plt.tight_layout()
    plt.show()
    plt.savefig(output_path)


# Captions on the validation set
attention_feats_shape = 64

for i in range(20):
    random_id = np.random.randint(0, len(images_val))
    print(f'{i}. {random_id}')
    image = images_val[random_id]
    actual_caption = ' '.join([tokenizer.index_word[i]
                               for i in caption_val[random_id]
                               if i != 0])
    actual_caption = (actual_caption
                      .replace('<start>', '')
                      .replace('<end>', ''))
    result, attention_plot = evaluate(image_captioner.encoder,
                                      image_captioner.decoder,
                                      tokenizer,
                                      image,
                                      max_length,
                                      attention_feats_shape)

    predicted_caption = (' '.join(result)
                         .replace('<start>', '')
                         .replace('<end>', ''))
    print(f'Actual caption: {actual_caption}')
    print(f'Predicted caption: {predicted_caption}')
    output_path = f'./{i}_attention_plot.png'
    plot_attention(image, result, attention_plot, output_path)
    print('------')
