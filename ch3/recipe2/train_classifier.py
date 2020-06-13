import pathlib

import h5py
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report

dataset_path = str(pathlib.Path.home() / '.keras' / 'datasets' /
                   'car_ims_rotated' / 'features.hdf5')
db = h5py.File(dataset_path, 'r')
SUBSET_INDEX = int(db['labels'].shape[0] * 0.5)
features = db['features'][:SUBSET_INDEX]
labels = db['labels'][:SUBSET_INDEX]

TRAIN_PROPORTION = 0.8
SPLIT_INDEX = int(len(labels) * TRAIN_PROPORTION)

X_train, y_train = (features[:SPLIT_INDEX],
                    labels[:SPLIT_INDEX])
X_test, y_test = (features[SPLIT_INDEX:],
                  labels[SPLIT_INDEX:])

model = LogisticRegressionCV(n_jobs=-1)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
report = classification_report(y_test, predictions,
                               target_names=db['label_names'])
print(report)
db.close()
