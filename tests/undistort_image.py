import pandas as pd
import numpy as np
import cv2 as cv
from datetime import datetime
import sys

class UndistortImage():
    '''
    undistorts an image using determined camera parameters.

    parameters:
        img (np.array): captured image
    '''

    def __init__(self, img):
        self.img = img

        # load config and read calibration directory
        self.config = pd.read_json('./.config.json')
        self.calib_dir = self.config.at['path_calibration', 'values']

        # prepare variables
        self.mtx = None
        self.new_mtx = None
        self.roi = None
        self.dist = None
        self.undist_img = None

        self.load_parameters()        
        self.undistort_image()
        self.save_image()
        

    def load_parameters(self):
        '''
        loads parameters from calibration dataframe
        '''

        try:
            calib_df = pd.read_json(f'{self.calib_dir}calibration.json')
        except FileNotFoundError:
            print('no calibration data found!')
            exit(-1)
        self.mtx = np.array(calib_df.at['camera_matrix', 'values'])
        self.new_mtx = np.array(calib_df.at['new_camera_matrix', 'values'])
        self.roi = np.array(calib_df.at['roi', 'values'])
        dist_k = calib_df.at['dist_k', 'values']
        dist_p = calib_df.at['dist_p', 'values']
        dist_s = calib_df.at['dist_s', 'values']
        dist_tau = calib_df.at['dist_tau', 'values']
        self.dist = np.array([dist_k[0], dist_k[1], dist_p[0], dist_p[1], dist_k[2], dist_k[3], dist_k[4], dist_k[5],
                        dist_s[0], dist_s[1], dist_s[2], dist_s[3], dist_tau[0], dist_tau[1]])


    def undistort_image(self):
        '''
        undistorts an image
        '''
        # Source:
        # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        
        self.img_undist = cv.undistort(self.img, self.mtx, self.dist, None, self.new_mtx)

        # crop image to ROI
        x, y, w, h = self.roi
        self.img_undist = self.img_undist[y:y+h, x:x+w]

    def save_image(self):
        '''
        saves image with a unique filename
        '''

        dt = datetime.now()
        cv.imwrite(f'image_{dt.year}{dt.month:02d}{dt.day:02d}_{dt.hour:02d}{dt.minute:02d}{dt.second:02d}.png', self.img_undist)



if __name__ == '__main__':
    print('use main.py')
    #exit(-1)
    sys.exit()