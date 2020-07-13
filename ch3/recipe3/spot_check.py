import json
import os
import pathlib
from glob import glob

import h5py
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *
from tqdm import tqdm

from ch3.recipe1.feature_extractor import FeatureExtractor

INPUT_SIZE = (224, 224, 3)

def get_pretrained_networks():
    return [
        (VGG16(input_shape=INPUT_SIZE,
               weights='imagenet',
               include_top=False),
         7 * 7 * 512),
        (VGG19(input_shape=INPUT_SIZE,
               weights='imagenet',
               include_top=False),
         7 * 7 * 512),
        (Xception(input_shape=INPUT_SIZE,
                  weights='imagenet',
                  include_top=False),
         7 * 7 * 2048),
        (ResNet152V2(input_shape=INPUT_SIZE,
                     weights='imagenet',
                     include_top=False),
         7 * 7 * 2048),
        (InceptionResNetV2(input_shape=INPUT_SIZE,
                           weights='imagenet',
                           include_top=False),
         5 * 5 * 1536)
    ]


def get_classifiers():
    models = {}
    models['LogisticRegression'] = LogisticRegression()
    models['SGDClf'] = SGDClassifier()
    models['PAClf'] = PassiveAggressiveClassifier()
    models['DecisionTreeClf'] = DecisionTreeClassifier()
    models['ExtraTreeClf'] = ExtraTreeClassifier()

    n_trees = 100
    models[f'AdaBoostClf-{n_trees}'] = \
        AdaBoostClassifier(n_estimators=n_trees)
    models[f'BaggingClf-{n_trees}'] = \
        BaggingClassifier(n_estimators=n_trees)
    models[f'RandomForestClf-{n_trees}'] = \
        RandomForestClassifier(n_estimators=n_trees)
    models[f'ExtraTreesClf-{n_trees}'] = \
        ExtraTreesClassifier(n_estimators=n_trees)
    models[f'GradientBoostingClf-{n_trees}'] = \
        GradientBoostingClassifier(n_estimators=n_trees)

    number_of_neighbors = range(3, 25)
    for n in number_of_neighbors:
        models[f'KNeighborsClf-{n}'] = \
            KNeighborsClassifier(n_neighbors=n)

    reg = [1e-3, 1e-2, 1, 10]
    for r in reg:
        models[f'LinearSVC-{r}'] = LinearSVC(C=r)
        models[f'RidgeClf-{r}'] = RidgeClassifier(alpha=r)

    print(f'Defined {len(models)} models.')
    return models


dataset_path = (pathlib.Path.home() / '.keras' / 'datasets' /
                'flowers17')
files_pattern = (dataset_path / 'images' / '*' / '*.jpg')
images_path = [*glob(str(files_pattern))]

labels = []
for index in tqdm(range(len(images_path))):
    image_path = images_path[index]
    image = load_img(image_path)

    label = image_path.split(os.path.sep)[-2]
    labels.append(label)

    image.close()

final_report = {}
best_model = None
best_accuracy = -1
best_features = None
for model, feature_size in get_pretrained_networks():
    output_path = dataset_path / f'{model.name}_features.hdf5'
    output_path = str(output_path)
    fe = FeatureExtractor(model=model,
                          input_size=INPUT_SIZE,
                          label_encoder=LabelEncoder(),
                          num_instances=len(images_path),
                          feature_size=feature_size,
                          output_path=output_path)

    fe.extract_features(image_paths=images_path,
                        labels=labels)

    db = h5py.File(output_path, 'r')

    TRAIN_PROPORTION = 0.8
    SPLIT_INDEX = int(len(labels) * TRAIN_PROPORTION)

    X_train, y_train = (db['features'][:SPLIT_INDEX],
                        db['labels'][:SPLIT_INDEX])
    X_test, y_test = (db['features'][SPLIT_INDEX:],
                      db['labels'][SPLIT_INDEX:])

    classifiers_report = {
        'extractor': model.name
    }

    print(f'Spot-checking with features from {model.name}')
    for clf_name, clf in get_classifiers().items():
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            print(f'\t{clf_name}: {e}')
            continue

        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print(f'\t{clf_name}: {accuracy}')
        classifiers_report[clf_name] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf_name
            best_features = model.name

    final_report[output_path] = classifiers_report
    db.close()

final_report['best_model'] = best_model
final_report['best_accuracy'] = best_accuracy
final_report['best_features'] = best_features

with open('final_report.json', 'w') as f:
    json.dump(final_report, f)
