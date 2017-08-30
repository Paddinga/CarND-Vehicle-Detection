import cv2
import numpy as np
from helper import *
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import time
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

# Read folder and create lists for images with cars and with no cars
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

# Read image and return features (spatial, histogram and HOG)
def image_features(image_source, color_space='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8,
                   cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
    # Empty list for features
    feature_list = []
    # Read image
    image = cv2.imread(image_source)
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
        feature_list.extend(spatial_features)
    # Add histogram features to list if True
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        feature_list.extend(hist_features)
    # Add HOG features to list if True
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog1 = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog2 = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog3 = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)
            hog_feat1 = hog1[:,:].ravel()
            hog_feat2 = hog2[:,:].ravel()
            hog_feat3 = hog3[:,:].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient, pix_per_cell,
                                            cell_per_block, vis=False, feature_vec=False)
        feature_list.extend(hog_features)
    # Return concatenated feature list
    return feature_list

def train_features(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                   spatial_feat, hist_feat, hog_feat):
    path = './data/'
    cars, nocars = load_data(path)
    print('Loaded images:', len(cars), 'with cars and', len(nocars), 'without cars')
    car_features = []
    nocar_features = []
    for element in cars:
        car_features.append(image_features(element, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block, hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat))
    for element in nocars:
        nocar_features.append(image_features(element, color_space=color_space, spatial_size=spatial_size,
                                             hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell,
                                             cell_per_block=cell_per_block, hog_channel=hog_channel,
                                             spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat))
    X = np.vstack((car_features, nocar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(nocar_features))))
    rnd_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rnd_state)
    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    return svc, X_scaler


def find_cars(image, ystart, ystop, scale, X_scaler, svc, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    bbox_list = []
    image_tosearch = image[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(image_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    for xb in range(nxsteps):
        for yb in range(nysteps):
            features = []
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            test_prediction = svc.predict(test_features)
            #test_prediction = 1
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bbox_list.append([(xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)])
    return bbox_list

class Heatmap:
    def __init__(self, buffer_length=5, threshold=5):
        self.buffer_length  = buffer_length
        self.threshold      = threshold
        self.heatmap_now    = np.array([[],[]])
        self.heatmap_out    = np.array([[],[]])
        self.heatmap_save   = np.array([[],[]])

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        self.heatmap_now = np.zeros((720, 1280)).astype(np.uint8)
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap_now[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        self.buffer_heat()

    def buffer_heat(self):
        if self.heatmap_save.size > 0:
            self.heatmap_save = np.vstack((self.heatmap_save, self.heatmap_now.reshape(1, 720, 1280)))
        else:
            self.heatmap_save = self.heatmap_now.reshape(1, 720, 1280)
        if self.heatmap_save.shape[0] > self.buffer_length:
            self.heatmap_save = self.heatmap_save[1:,:,:]
        self.heatmap_out = np.sum(self.heatmap_save, axis=0)
        self.heatmap_out[self.heatmap_out <= self.threshold] = 0
        self.heatmap_out = np.clip(self.heatmap_out, 0, 255)

def pipeline(image):
    bbox_list_1 = find_cars(image, ystart, ystop, 1, X_scaler, svc, orient, pix_per_cell, cell_per_block,
                            spatial_size, hist_bins)
    bbox_list_2 = find_cars(image, ystart, ystop, 1.5, X_scaler, svc, orient, pix_per_cell, cell_per_block,
                          spatial_size, hist_bins)
    bbox_list_3 = find_cars(image, ystart, ystop, 2, X_scaler, svc, orient, pix_per_cell, cell_per_block,
                            spatial_size, hist_bins)
    heatmap.add_heat(bbox_list_1 + bbox_list_2 + bbox_list_3)
    labels = label(heatmap.heatmap_out)
    out_image = draw_labeled_bboxes(image, labels)
    return out_image

### MAIN ###

# Parameters
color_space     = 'YCrCb'   # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient          = 12        # HOG orientations
pix_per_cell    = 16        # HOG pixels per cell
cell_per_block  = 2         # HOG cells per block
hog_channel     = 'ALL'     # Can be 0, 1, 2, or "ALL"
spatial_size    = (16, 16)  # Spatial binning dimensions
hist_bins       = 16        # Number of histogram bins
spatial_feat    = True      # Spatial features on or off
hist_feat       = True      # Histogram features on or off
hog_feat        = True      # HOG features on or off
ystart          = 400
ystop           = 650

svc, X_scaler = train_features(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins,
                               spatial_feat, hist_feat, hog_feat)
heatmap = Heatmap(30, 40)
output = 'project_video_out.mp4'
clip1 = VideoFileClip('project_video.mp4')
clip = clip1.fl_image(pipeline)
clip.write_videofile(output, audio=False)
