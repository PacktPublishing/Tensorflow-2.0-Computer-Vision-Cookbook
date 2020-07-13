import pathlib

import h5py
from creme import stream
from creme.linear_model import LogisticRegression
from creme.metrics import Accuracy
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler


def write_dataset(output_path, feats, labels, batch_size):
    feature_size = feats.shape[1]
    csv_columns = ['class'] + [f'feature_{i}'
                               for i in range(feature_size)]

    dataset_size = labels.shape[0]
    with open(output_path, 'w') as f:
        f.write(f'{",".join(csv_columns)}\n')

        for batch_number, index in \
                enumerate(range(0, dataset_size, batch_size)):
            print(f'Processing batch {batch_number + 1} of '
                  f'{int(dataset_size / float(batch_size))}')

            batch_feats = feats[index: index + batch_size]
            batch_labels = labels[index: index + batch_size]

            for label, vector in \
                    zip(batch_labels, batch_feats):
                vector = ','.join([str(v) for v in vector])
                f.write(f'{label},{vector}\n')


dataset_path = str(pathlib.Path.home() / '.keras' / 'datasets' /
                   'car_ims_rotated' / 'features.hdf5')
db = h5py.File(dataset_path, 'r')

TRAIN_PROPORTION = 0.8
SPLIT_INDEX = int(db['labels'].shape[0] * TRAIN_PROPORTION)

BATCH_SIZE = 256
write_dataset('train.csv',
              db['features'][:SPLIT_INDEX],
              db['labels'][:SPLIT_INDEX],
              BATCH_SIZE)
write_dataset('test.csv',
              db['features'][SPLIT_INDEX:],
              db['labels'][SPLIT_INDEX:],
              BATCH_SIZE)

FEATURE_SIZE = db['features'].shape[1]
types = {f'feature_{i}': float for i in range(FEATURE_SIZE)}
types['class'] = int

model = StandardScaler()
model |= OneVsRestClassifier(LogisticRegression())

metric = Accuracy()
dataset = stream.iter_csv('train.csv',
                          target_name='class',
                          converters=types)
print('Training started...')
for i, (X, y) in enumerate(dataset):
    predictions = model.predict_one(X)
    model = model.fit_one(X, y)
    metric = metric.update(y, predictions)

    if i % 100 == 0:
        print(f'Update {i} - {metric}')

print(f'Final - {metric}')

metric = Accuracy()
test_dataset = stream.iter_csv('test.csv',
                               target_name='class',
                               converters=types)
print('Testing model...')
for i, (X, y) in enumerate(test_dataset):
    predictions = model.predict_one(X)
    metric = metric.update(y, predictions)

    if i % 1000 == 0:
        print(f'(TEST) Update {i} - {metric}')

print(f'(TEST) Final - {metric}')
