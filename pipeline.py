import cv2
import numpy as np
from helper import *
import glob

def image_features(image, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                   cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Empty list for features
    feature_list = []
    # Color conversion to color_space
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    # Add spatial features to list if True
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        feature_list.append(spatial_features)
    # Add histogram features to list if True
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        feature_list.append(hist_features)
    # Add HOG features to list if True
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell,
                                                     cell_per_block, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,pix_per_cell,
                                            cell_per_block, vis=False, feature_vec=True)
            feature_list.append(hog_features)
    # Return concatenated feature list
    return np.concatenate(feature_list)

def load_data(path):
    images = glob.glob(path + '*.png')
    cars =[]
    nocars = []
    for image in images:
        if 'image' in image or 'extra' in image:
            nocars.append(image)
        else:
            cars.append(image)
    return cars, nocars


path = './data/'
load_data(path)


