import os
import random

import librosa
import numpy as np
from tqdm import tqdm


class FeatureExtractor(object):
    """return X_features"""

    def __init__(self, rate):
        self.rate = rate

    def get_features(self, features_to_use, X, n_mfcc=13):
        X_features = None
        accepted_features_to_use = ("mfcc-on-data", "mfcc-on-feature",)
        if features_to_use not in accepted_features_to_use:
            raise NotImplementedError("{} not in {}!".format(features_to_use, accepted_features_to_use))
        if features_to_use in ('mfcc-on-data',):
            X_features = self.get_mfcc_across_data(X, n_mfcc=n_mfcc)

        elif features_to_use in ('mfcc-on-feature',):
            X_features = self.get_mfcc_across_features(X, n_mfcc=n_mfcc)
        return X_features

    def get_mfcc_across_data(self, X, n_mfcc=13):
        """get mean of mfcc features across frame"""
        print("building mfcc features...")
        X_features = np.apply_along_axis(lambda x: np.mean(librosa.feature.mfcc(x, sr=self.rate,
                                                                                n_mfcc=n_mfcc),
                                                           axis=0), 1, X)
        return X_features

    def get_mfcc_across_features(self, X, n_mfcc=13):
        """get mean, variance, max, min of mfcc features across feature"""

        def _get_mfcc_features(x):
            mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
            mean = np.mean(mfcc_data, axis=1)
            var = np.var(mfcc_data, axis=1)
            maximum = np.max(mfcc_data, axis=1)
            minimum = np.min(mfcc_data, axis=1)
            out = np.array(list(mean) + list(var) + list(maximum) + list(minimum))
            return out

        X_features = np.apply_along_axis(_get_mfcc_features, 1, X)
        return X_features

    # def get_mfcc_across_features(self, X, n_mfcc=13):
    #     """get mean, variance, max, min of mfcc features across feature"""
    #
    #     def _get_mfcc_features(x):
    #         mfcc_data = librosa.feature.mfcc(x, sr=self.rate, n_mfcc=n_mfcc)
    #         mean = np.mean(mfcc_data, axis=1)
    #         var = np.var(mfcc_data, axis=1)
    #         maximum = np.max(mfcc_data, axis=1)
    #         minimum = np.min(mfcc_data, axis=1)
    #         out = np.array(list(mean) + list(var) + list(maximum) + list(minimum))
    #         return out
    #
    #     path = ('noises/')
    #     noise = os.listdir(path)
    #     for i in range(len(noise)):
    #         noise[i] = os.path.join(path, noise[i])
    #     for i in tqdm(range(len(X))):
    #         t = random.randint(0, 9)
    #         if (t < 3):
    #             t = random.randint(0, len(noise) - 1)
    #             n, _ = librosa.load(noise[t], sr=16000)
    #             X[i] = X[i] + n[0:len(X[i] - 1)]
    #     X_features = np.apply_along_axis(_get_mfcc_features, 1, X)
    #     return X_features
