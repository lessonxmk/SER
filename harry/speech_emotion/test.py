import glob
import os
import pickle

import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from speech_emotion.features import FeatureExtractor
from speech_emotion.models import KerasModel
from train_2 import process_data_ravdess

MODEL_NAME = "experiments/16000_ravdess_2_mfcc-on-feature_cnn_model_417_0.hdf5"

PATH = "E:/Test/Session1-4/noises/n02/"
RATE = 16000
FEATURES_TO_USE = "mfcc-on-feature"
TEST_FEATURES = False


def process_data_ravdess(path, t=2, n_samples=3):
    """construct dataset X, y from yscz dataset, each wavefile is sampled
        n_samples times at t seconds; Meanwhile, a meta dictionary is built which has
         key: wavefile basename, values: {'X', 'y', 'path'}; 'X' is stacked numpy array
         of normalized wave amplitude samples, y is a list of string labels,
         path is path/to/wavefile"""
    path = path.rstrip('/')
    wav_files = glob.glob(path + '/*.wav')
    meta_dict = {}

    LABEL_DICT1 = {
        '01': 'neutral',
        '02': 'frustration',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'excitement',
        '08': 'surprised'
    }
    LABEL_DICT2 = {
        '01': 'neutral',
        '02': 'negative',
        '03': 'active',
        '04': 'negative',
        '05': 'negative',
        '06': 'negative',
        '07': 'active',
        '08': 'neutral'
    }

    print("constructing meta dictionary for {}...".format(path))
    for i, wav_file in enumerate(tqdm(wav_files)):
        label = LABEL_DICT1[str(os.path.basename(wav_file).split('-')[2])]
        # label = LABEL_DICT2[str(os.path.basename(wav_file).split('-')[2])]
        wav_data, _ = librosa.load(wav_file, sr=RATE)
        X1 = []
        y1 = []
        if (len(wav_data) - t * RATE <= 0):
            continue
        for index in np.random.choice(range(len(wav_data) - t * RATE), n_samples, replace=False):
            X1.append(wav_data[index: (index + t * RATE)])
            y1.append(label)
        X1 = np.array(X1)
        meta_dict[i] = {
            'X': X1,
            'y': y1,
            'path': wav_file
        }

    print("building X, y...")
    X = []
    y = []
    for k in meta_dict:
        X.append(meta_dict[k]['X'])
        y += meta_dict[k]['y']
    X = np.row_stack(X)
    y = np.array(y)
    assert len(X) == len(y), "X length and y length must match! X shape: {}, y length: {}".format(X.shape, y.shape)
    return X, y
if (TEST_FEATURES == True):
    if (FEATURES_TO_USE == "mfcc-on-feature"):
        with open('test_on_feature.pkl', 'rb')as f:
            features = pickle.load(f)
    elif (FEATURES_TO_USE == "mfcc-on-data"):
        with open('test_on_data.pkl', 'rb')as f:
            features = pickle.load(f)
    X_features = features['X']
    y = features['y']
    lb_encoder = features['lb_encoder']
    n_class = len(lb_encoder.classes_)
else:
    X, y = process_data_ravdess(PATH, 2)
    lb_encoder = LabelEncoder()
    y = lb_encoder.fit_transform(y)
    print(X.shape)
    print(y.shape)
    feature_extractor = FeatureExtractor(rate=RATE)
    X_features = feature_extractor.get_features(FEATURES_TO_USE, X, n_mfcc=13)
    features = {'X': X_features, 'y': y,
                    'lb_encoder':lb_encoder}
    if (FEATURES_TO_USE == "mfcc-on-feature"):
        with open('test_on_feature.pkl', 'wb') as f:
            pickle.dump(features, f)
    elif (FEATURES_TO_USE == "mfcc-on-data"):
        with open('test_on_data.pkl', 'wb') as f:
            pickle.dump(features, f)

model = KerasModel()
model.load(MODEL_NAME)

pred_valid_y = model.predict(X_features)
print("accuracy on test set is {}...".format(accuracy_score(y, pred_valid_y)))
cnf_matrix = confusion_matrix(y, pred_valid_y)
print(cnf_matrix)
print("class names are {}...".format(lb_encoder.classes_))
