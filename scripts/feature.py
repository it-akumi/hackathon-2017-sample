import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../agent')
import numpy as np
import six.moves.cPickle as pickle

from PIL import Image
from PIL import ImageOps

from ml.cnn_feature_extractor import CnnFeatureExtractor

def observation(rgb_file_name, depth_file_name, depth_image_dim=32*32):
    image = []
    image.append(Image.open(rgb_file_name))

    depth = []
    d = Image.open(depth_file_name)
    depth.append(np.array(ImageOps.grayscale(d)).reshape(depth_image_dim))

    observation = {"image": image, "depth": depth}

    return observation


def main():
    ob = observation(sys.argv[1], sys.argv[2])
    print('load pickle')
    feature_extractor = pickle.load(open(os.path.dirname(os.path.abspath(__file__)) + '/../agent/model/alexnet_feature_extractor.pickle'))
    print('done load pickle')
    obs_array = feature_extractor.feature(ob, 1)
    print(len(obs_array))
    print(obs_array)


if __name__ == '__main__':
    main()

