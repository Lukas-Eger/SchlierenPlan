
import sys
import cv2 as cv
from vimba import *
import os
import pandas as pd
import numpy as np
import time
import math
import subprocess
from pynput import keyboard
from collections import namedtuple
from classes import undistort_image
import platform

from classes.windows_path import windows_path

class CaptureImages():
    '''
    class for image capturing

    parameters:
        preview (bool): start in preview mode.
        pattern (str): pattern to display on screen. \'None\' only captures an image without showing a pattern. valid inputs:\'active_checkerboard\', \'phase_shift_pattern\', \'test_pattern\'.
        pose_distance (str): distance to calibration target in cm.
        pose_comment (str): comment to describe captures pose.
        wait (bool): waiting time before image capturing.
        preview_options (str): configures preview mode. \'1\': hide menu; \'2\': show screen border; \'3\' show checkerboard border. multiple options at once are possible, e.g. to hide the menu and show the checkerboard border use \'13\'.
        undistort (bool): capture image and start undistort_image.py
    '''
    def __init__(self, preview:bool, pattern:str = None, pose_distance:str = None, pose_comment:str = None, wait:bool = True, preview_options:str = '', undistort:bool=False):
        # initial values
        self.preview = preview
        self.undistort = undistort
        self.pattern = pattern
        self.wait = wait

        self.is_linux = True if platform.system() == 'Linux' else False
        self.coordinates = namedtuple('coordinates', ['x', 'y'])

        # configure preview
        self.stop_preview = False
        self.hide_menu = False if not '1' in preview_options else True
        self.show_screen_border = False if not '2' in preview_options else True
        self.show_checkerboard_border = False if not '3' in preview_options else True
        self.border_thickness = 5


        # configure camera and screen
        self.config = pd.read_json('./.config.json')
        self.exposure_time = self.config.at['exposure_time_us', 'values']
        self.gain = self.config.at['gain', 'values']
        resolution = self.config.at['resolution', 'values']
        self.res_x = int(resolution.split('x')[0])
        self.res_y = int(resolution.split('x')[1])


        # check if directory exists, create if not
        self.poses_path = self.config.at['path_poses', 'values']
        if not os.path.isdir(self.poses_path):
            if self.is_linux:
                path = self.poses_path
            else:
                path = windows_path(self.poses_path)
            subprocess.run(f'mkdir {path}', shell = True, check = True, text = True)

        if self.preview or self.undistort:
            self.capture()
        elif self.undistort:
            self.capture()

        # create unique pose name
        else:
            self.pose_name = self.pattern
            if pose_comment:
                self.pose_name += f'_{pose_comment}'
            if pose_distance:
                self.pose_name += f'_distance_{pose_distance}_cm'
            # check if pose name is unique
            index = 0
            while True:
                if not os.path.isdir(self.poses_path + self.pose_name + f'_{index}'):
                    self.pose_name += f'_{index}/'
                    break
                else:
                    index += 1
            # create directory
            if self.is_linux:
                path = self.poses_path + self.pose_name
            else:
                path = windows_path(self.poses_path + self.pose_name)
            subprocess.run(f'mkdir {path}', shell = True, check = True, text = True)
                
            if pattern == 'active_checkerboard':
                self.img_path = self.config.at['path_checkerboard_pattern', 'values']
                # copy parameters to captured pose
                if self.is_linux:
                    subprocess.run(f'cp {self.img_path}param.json {self.poses_path + self.pose_name}', shell = True, check = True, text = True)
                else:
                    path_1 = windows_path(f'{self.img_path}param.json')
                    path_2 = windows_path(self.poses_path + self.pose_name)
                    subprocess.run(f'copy {path_1} {path_2}', shell = True, check = True, text = True)
            elif pattern == 'phase_shift':
                self.img_path = self.config.at['path_phase_shift_pattern', 'values']
                # copy parameters to captured pose
                if self.is_linux:
                    subprocess.run(f'cp {self.img_path}param.json {self.poses_path + self.pose_name}', shell = True, check = True, text = True)
                else:
                    path_1 = windows_path(f'{self.img_path}param.json')
                    path_2 = windows_path(self.poses_path + self.pose_name)
                    subprocess.run(f'copy {path_1} {path_2}', shell = True, check = True, text = True)
            elif pattern == 'test_pattern':
                self.img_path = self.config.at['path_test_pattern', 'values']
            else:
                self.img_path = None
            if self.img_path and not os.path.isdir(self.img_path) and not self.preview:
                print('no pattern found. returning to main menu...')
                time.sleep(2)
                return

            self.capture()
            
    
    def capture(self):
        '''
        main function for image capturing
        '''

        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            if len(cams) == 0:
                print('no camera found')
                #exit(-1)
                sys.exit()
            with cams[0] as cam:
                # exposure time and gain from config
                cam.ExposureTime.set(self.exposure_time)
                cam.Gain.set(self.gain)

                # use auto exposure time mode and update config
                if self.preview or self.undistort:
                    cam.ExposureAuto = True
                    last_exposure_time = self.exposure_time
                    counter = 0
                    while True:
                        frame = cam.get_frame()
                        exposure_time = cam.ExposureTime.get()
                        print(f'\x1b[K exposure time: {exposure_time}us', end='\r')
                        if exposure_time == last_exposure_time:
                            if counter == 2:
                                self.config.at['exposure_time_us', 'values'] = exposure_time
                                break
                            else:
                                counter += 1
                        last_exposure_time = exposure_time

                    print(f'\x1b[Kexposure time: {exposure_time}us')
                    cam.ExposureAuto = False
                    
                    # show preview
                    cv.namedWindow('screen', cv.WND_PROP_FULLSCREEN)
                    cv.setWindowProperty('screen', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

                    # start keyboard listener
                    listener = keyboard.Listener(
                    on_press=self.on_press)
                    listener.start()

                    # load checkerboard and extract border
                    checkerboard = cv.imread(self.config.at['path_checkerboard_pattern', 'values'] + 'checkerboard.png')
                    checkerboard = cv.cvtColor(checkerboard, cv.COLOR_BGR2GRAY)
                    checkerboard_img_res = self.coordinates(np.shape(checkerboard)[1], np.shape(checkerboard)[0])
                    
                    # find border
                    x_left, x_right, y_upper, y_lower = None, None, None, None
                    # find left border
                    for x in range(checkerboard_img_res.x):
                        if checkerboard[int(checkerboard_img_res.y / 2)][x] == 255:
                            x_left = x
                            break
                    # find right border
                    for x in reversed(range(checkerboard_img_res.x)):
                        if checkerboard[int(checkerboard_img_res.y / 2)][x] == 255:
                            x_right = x
                            break
                    # find upper border
                    for y in range(checkerboard_img_res.y):
                        if checkerboard[y][int(checkerboard_img_res.x / 2)] == 255:
                            y_upper = y
                            break
                    # find lower border
                    for y in reversed(range(checkerboard_img_res.y)):
                        if checkerboard[y][int(checkerboard_img_res.x / 2)] == 255:
                            y_lower = y
                            break
                    
                    # save coordinates of checkerboard upper left corner and board size
                    c_upper_left_corner = self.coordinates(x_left, y_upper)
                    c_size = self.coordinates(x_right - x_left, y_lower - y_upper)

                    # start preview
                    while not self.stop_preview:
                        cam.Gain.set(self.gain)

                        # get current frame and flip image
                        frame = cam.get_frame()
                        frame.convert_pixel_format(PixelFormat.Mono8)
                        frame = frame.as_opencv_image()
                        frame = np.array(frame)
                        frame = np.flip(frame)

                        # scale to height of screen
                        scaling = self.res_y / np.shape(frame)[0]
                        new_x = int(np.shape(frame)[1] * scaling)
                        new_y = int(np.shape(frame)[0] * scaling)
                        frame = cv.resize(frame, [new_x, new_y])

                        # add padding left and right
                        padding = math.floor((self.res_x - new_x) / 2) - self.border_thickness
                        new_frame = []
                        for i, line in enumerate(frame):
                            # left padding
                            new_line = [0 for p in range(padding)]
                            new_line += [50 for p in range(self.border_thickness)]
                            new_line += [p for p in line]
                            # right padding
                            new_line += [50 for p in range(self.border_thickness)]
                            new_line += [0 for p in range(padding)]
                            new_frame.append(new_line)
                        new_frame = np.array(new_frame, dtype=np.uint8)

                        # add borders if selected
                        if self.show_screen_border:
                            cv.rectangle(new_frame, (int(self.border_thickness / 2), int(self.border_thickness / 2)), 
                                    (self.res_x - int(self.border_thickness / 2) - 1, self.res_y - int(self.border_thickness / 2) - 1), 
                                    255, self.border_thickness, cv.LINE_8)
                        elif self.show_checkerboard_border:
                            cv.rectangle(new_frame, (c_upper_left_corner.x, c_upper_left_corner.y), (c_upper_left_corner.x + c_size.x, c_upper_left_corner.y + c_size.y), 255, self.border_thickness)

                        # add text to preview
                        text_height = cv.getTextSize('I', cv.FONT_HERSHEY_COMPLEX, 2, 1)[0][1]
                        space = 5
                        text = []
                        if not self.undistort:
                            text += ['[1] hide menu'] if not self.hide_menu else ['[1] show menu']
                            text += ['[2] hide screen border'] if self.show_screen_border else ['[2] show screen border']
                            text += ['[3] hide checkerboard border'] if self.show_checkerboard_border else ['[3] show checkerboard border']
                        text += ['', f'current Gain: {self.gain}dB']
                        if self.gain < 27:
                            text += ['[w] increase Gain']
                        if self.gain > 0:
                            text += ['[s] decrease Gain']
                        if not self.undistort:
                            text += ['', '[space] stop preview']
                        else:
                            text += ['', '[space] capture image']
                        if not self.hide_menu:
                            for i, t in enumerate(text):
                                y_pos = self.border_thickness + (i + 1) * space + (i + 1) * text_height
                                x_pos = self.border_thickness + space
                                cv.putText(new_frame, t, (x_pos, y_pos), cv.FONT_HERSHEY_COMPLEX, 1, 150)

                        cv.imshow('screen', new_frame)
                        cv.waitKey(1)
                    cv.destroyAllWindows()

                    if self.undistort:
                        # get current frame and flip image
                        frame = cam.get_frame()
                        frame.convert_pixel_format(PixelFormat.Mono8)
                        frame = frame.as_opencv_image()
                        frame = np.array(frame)
                        frame = np.flip(frame)

                        undistort_image.UndistortImage(frame)

                    # update config
                    self.config.at['gain', 'values'] = self.gain
                    self.config.to_json('./.config.json')

                elif not self.img_path: # passive targets
                    if self.wait:
                        for i in reversed(range(3)):
                            print(f'\x1b[K image capturing starts in {i + 1}s', end='\r')  
                            time.sleep(1)

                    # get current frame and flip image
                    frame = cam.get_frame()
                    frame.convert_pixel_format(PixelFormat.Mono8)
                    frame = frame.as_opencv_image()
                    frame = np.array(frame)
                    frame = np.flip(frame)

                    # save image
                    cv.imwrite(self.poses_path + self.pose_name + 'checkerboard.png', frame)
                    print(f'pose saved at: {self.poses_path + self.pose_name}')
                    time.sleep(2)
                else:
                    if self.wait:
                        for i in reversed(range(3)):
                            print(f'\x1b[K image capturing starts in {i + 1}s', end='\r')  
                            time.sleep(1)

                    # get all patterns from pattern directory
                    patterns = [p for p in os.listdir(self.img_path) if p.split('.')[-1] == 'png']
                    patterns.sort()
                    cv.namedWindow('screen', cv.WND_PROP_FULLSCREEN)
                    cv.setWindowProperty('screen', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

                    # show patterns and capture image
                    for ip, pattern in enumerate(patterns):
                        if len(patterns) > 1:
                            print(f'\x1b[K capturing images...{ip + 1}/{len(patterns)} ({round(ip/(len(patterns) - 1) * 100, 2)}%)', end='\r')
                        img = cv.imread(self.img_path + pattern)
                        cv.imshow('screen', img) # show pattern
                        cv.waitKey(1)
                        time.sleep(.1)
                        frame = cam.get_frame() # capture image
                        frame.convert_pixel_format(PixelFormat.Mono8)
                        frame = frame.as_opencv_image()
                        frame = np.array(frame)
                        frame = np.flip(frame)
                        cv.imwrite(self.poses_path + self.pose_name + pattern, frame)
                    print(f'\x1b[Kcapturing images...done')

                    print(f'pose saved at: {self.poses_path + self.pose_name}')
                    cv.destroyAllWindows()
                    if self.wait:
                        time.sleep(2)

    def on_press(self, key):
        '''
        function for keyboard listener
        '''

        try:
            if key.char == '1': # hide/show menu
                self.hide_menu = not self.hide_menu
            elif key.char == '2': # hide/show screen border
                self.show_screen_border = not self.show_screen_border
                self.show_checkerboard_border = False
            elif key.char == '3': # hide/show checkerboard border
                self.show_checkerboard_border = not self.show_checkerboard_border
                self.show_screen_border = False
            elif key.char == 'w': # increase gain
                self.gain += .5
                if self.gain > 27:
                    self.gain = 27.0
            elif key.char == 's': # decrease gain
                self.gain -= .5
                if self.gain < 0:
                    self.gain = 0
        except AttributeError:
            if key == keyboard.Key.space: # stop preview
                self.stop_preview = True


if __name__ == '__main__':
    print('use main.py')
    #exit(-1)
    sys.exit()
    
    
    
    
#import cv2 as cv
# import os
# import pandas as pd
# import numpy as np
# import time
# import math
# import subprocess
# from pynput import keyboard
# from collections import namedtuple
# from classes import undistort_image
# import platform
# import sys

# from classes.windows_path import windows_path

# class CaptureImages():
#     '''
#     class for image capturing

#     parameters:
#         preview (bool): start in preview mode.
#         pattern (str): pattern to display on screen. \'None\' only captures an image without showing a pattern. valid inputs:\'active_checkerboard\', \'phase_shift_pattern\', \'test_pattern\'.
#         pose_distance (str): distance to calibration target in cm.
#         pose_comment (str): comment to describe captures pose.
#         wait (bool): waiting time before image capturing.
#         preview_options (str): configures preview mode. \'1\': hide menu; \'2\': show screen border; \'3\' show checkerboard border. multiple options at once are possible, e.g. to hide the menu and show the checkerboard border use \'13\'.
#         undistort (bool): capture image and start undistort_image.py
#     '''
#     def __init__(self, preview:bool, pattern:str = None, pose_distance:str = None, pose_comment:str = None, wait:bool = True, preview_options:str = '', undistort:bool=False):
#         # initial values
#         self.preview = preview
#         self.undistort = undistort
#         self.pattern = pattern
#         self.wait = wait

#         self.is_linux = True if platform.system() == 'Linux' else False
#         self.coordinates = namedtuple('coordinates', ['x', 'y'])

#         # configure preview
#         self.stop_preview = False
#         self.hide_menu = False if not '1' in preview_options else True
#         self.show_screen_border = False if not '2' in preview_options else True
#         self.show_checkerboard_border = False if not '3' in preview_options else True
#         self.border_thickness = 5


#         # configure camera and screen
#         self.config = pd.read_json('./.config.json')
#         self.exposure_time = self.config.at['exposure_time_us', 'values']
#         self.gain = self.config.at['gain', 'values']
#         resolution = self.config.at['resolution', 'values']
#         self.res_x = int(resolution.split('x')[0])
#         self.res_y = int(resolution.split('x')[1])


#         # check if directory exists, create if not
#         self.poses_path = self.config.at['path_poses', 'values']
#         if not os.path.isdir(self.poses_path):
#             if self.is_linux:
#                 path = self.poses_path
#             else:
#                 path = windows_path(self.poses_path)
#             subprocess.run(f'mkdir {path}', shell = True, check = True, text = True)

#         if self.preview or self.undistort:
#             self.capture()
#         elif self.undistort:
#             self.capture()

#         # create unique pose name
#         else:
#             self.pose_name = self.pattern
#             if pose_comment:
#                 self.pose_name += f'_{pose_comment}'
#             if pose_distance:
#                 self.pose_name += f'_distance_{pose_distance}_cm'
#             # check if pose name is unique
#             index = 0
#             while True:
#                 if not os.path.isdir(self.poses_path + self.pose_name + f'_{index}'):
#                     self.pose_name += f'_{index}/'
#                     break
#                 else:
#                     index += 1
#             # create directory
#             if self.is_linux:
#                 path = self.poses_path + self.pose_name
#             else:
#                 path = windows_path(self.poses_path + self.pose_name)
#             subprocess.run(f'mkdir {path}', shell = True, check = True, text = True)
                
#             if pattern == 'active_checkerboard':
#                 self.img_path = self.config.at['path_checkerboard_pattern', 'values']
#                 # copy parameters to captured pose
#                 if self.is_linux:
#                     subprocess.run(f'cp {self.img_path}param.json {self.poses_path + self.pose_name}', shell = True, check = True, text = True)
#                 else:
#                     path_1 = windows_path(f'{self.img_path}param.json')
#                     path_2 = windows_path(self.poses_path + self.pose_name)
#                     subprocess.run(f'copy {path_1} {path_2}', shell = True, check = True, text = True)
#             elif pattern == 'phase_shift':
#                 self.img_path = self.config.at['path_phase_shift_pattern', 'values']
#                 # copy parameters to captured pose
#                 if self.is_linux:
#                     subprocess.run(f'cp {self.img_path}param.json {self.poses_path + self.pose_name}', shell = True, check = True, text = True)
#                 else:
#                     path_1 = windows_path(f'{self.img_path}param.json')
#                     path_2 = windows_path(self.poses_path + self.pose_name)
#                     subprocess.run(f'copy {path_1} {path_2}', shell = True, check = True, text = True)
#             elif pattern == 'test_pattern':
#                 self.img_path = self.config.at['path_test_pattern', 'values']
#             else:
#                 self.img_path = None
#             if self.img_path and not os.path.isdir(self.img_path) and not self.preview:
#                 print('no pattern found. returning to main menu...')
#                 time.sleep(2)
#                 return

#             self.capture()
            
    
#     def capture(self):
#         '''
#         main function for image capturing
#         '''

#         cap = cv.VideoCapture(2, cv.CAP_DSHOW)  # Open the camera

#         if not cap.isOpened():
#             print('Camera could not be opened')
#             #exit(-1)
#             sys.exit()

#         #cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
#         #cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
#         cap.set(cv.CAP_PROP_EXPOSURE, self.exposure_time)
#         cap.set(cv.CAP_PROP_GAIN, self.gain)
        
#         # use auto exposure time mode and update config
#         if self.preview or self.undistort:
#             # show preview
#             cv.namedWindow('screen', cv.WND_PROP_FULLSCREEN)
#             cv.setWindowProperty('screen', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            
#             while not self.stop_preview:
#                 ret, frame = cap.read()
                
#                 # start keyboard listener
#                 listener = keyboard.Listener(on_press=self.on_press)
#                 listener.start()

#                 if not ret:
#                     print('Error: Frame could not be captured.')
#                     break

#                 # scale to fit the screen
#                 frame = cv.resize(frame, (self.res_x, self.res_y))

#                 # add borders if selected
#                 if self.show_screen_border:
#                     cv.rectangle(frame, (0, 0), (self.res_x - 1, self.res_y - 1), (255, 255, 255), self.border_thickness)
#                 elif self.show_checkerboard_border:
#                     # add code to draw checkerboard border
#                     pass

#                 # add text to preview
#                 text = ''
#                 if not self.undistort:
#                     text += '[1] hide menu' if not self.hide_menu else '[1] show menu'
#                     text += '[2] hide screen border' if self.show_screen_border else '[2] show screen border'
#                     text += '[3] hide checkerboard border' if self.show_checkerboard_border else '[3] show checkerboard border'
#                 text += f'current Gain: {self.gain}dB'
#                 if self.gain < 27:
#                     text += '[w] increase Gain'
#                 if self.gain > 0:
#                     text += '[s] decrease Gain'
#                 if not self.undistort:
#                     text += '[space] stop preview'
#                 else:
#                     text += '[space] capture image'

#                 if not self.hide_menu:
#                     cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (150, 150, 150), 2)

#                 cv.imshow('screen', frame)
#                 cv.waitKey(1)

#             cv.destroyAllWindows()

#             if self.undistort:
#                 # capture an image for undistortion
#                 ret, frame = cap.read()

#                 if not ret:
#                     print('Error: Frame could not be captured.')
#                     return

#                 # save captured image
#                 cv.imwrite(os.path.join(self.poses_path, self.pose_name + 'checkerboard.png'), frame)

#         elif not self.img_path:  # passive targets
#             if self.wait:
#                 for i in reversed(range(3)):
#                     print(f'image capturing starts in {i + 1}s', end='\r')
#                     time.sleep(1)

#             # capture an image
#             ret, frame = cap.read()

#             if not ret:
#                 print('Error: Frame could not be captured.')
#                 return

#             # save captured image
            
#             cv.imwrite(os.path.join(self.poses_path, self.pose_name + 'checkerboard.png'), frame)
#             print(f'pose saved at: {self.poses_path + self.pose_name}')
#             time.sleep(2)

#         else:
#             if self.wait:
#                 for i in reversed(range(3)):
#                     print(f'image capturing starts in {i + 1}s', end='\r')
#                     time.sleep(1)

#             # get all patterns from pattern directory
#             patterns = [p for p in os.listdir(self.img_path) if p.split('.')[-1] == 'png']
#             patterns.sort()
#             cv.namedWindow('screen', cv.WND_PROP_FULLSCREEN)
#             cv.setWindowProperty('screen', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

#             # show patterns and capture images
#             for ip, pattern in enumerate(patterns):
#                 if len(patterns) > 1:
#                     print(f'capturing images...{ip + 1}/{len(patterns)} ({round(ip/(len(patterns) - 1) * 100, 2)}%)', end='\r')
#                 img = cv.imread(os.path.join(self.img_path, pattern))
#                 cv.imshow('screen', img)  # show pattern
#                 cv.waitKey(1)
#                 time.sleep(.1)
#                 ret, frame = cap.read()  # capture image

#                 if not ret:
#                     print('Error: Frame could not be captured.')
#                     return

#                 # save captured image
#                 # frame = self.capture_frame_from_camera()
#                 # cv.imwrite(self.poses_path + self.pose_name + 'captured_image.png', frame)
#                 # print(f'Bild aufgenommen und gespeichert unter: {self.poses_path + self.pose_name}captured_image.png')
#                 cv.imwrite(self.poses_path + self.pose_name + 'checkerboard.png', frame)
#                 print(f'pose saved at: {self.poses_path + self.pose_name}')
#                 time.sleep(2)
               
#                 #cv.imwrite(os.path.join(self.poses_path, self.pose_name + pattern), frame)

#             print('capturing images...done')
#             print(f'pose saved at: {self.poses_path + self.pose_name}')
#             cv.destroyAllWindows()
#             if self.wait:
#                 time.sleep(2)

#         cap.release()

#     def on_press(self, key):
#         '''
#         function for keyboard listener
#         '''

#         try:
#             if key.char == '1':  # hide/show menu
#                 self.hide_menu = not self.hide_menu
#             elif key.char == '2':  # hide/show screen border
#                 self.show_screen_border = not self.show_screen_border
#                 self.show_checkerboard_border = False
#             elif key.char == '3':  # hide/show checkerboard border
#                 self.show_checkerboard_border = not self.show_checkerboard_border
#                 self.show_screen_border = False
#             elif key.char == 'w':  # increase gain
#                 self.gain += .5
#                 if self.gain > 27:
#                     self.gain = 27.0
#             elif key.char == 's':  # decrease gain
#                 self.gain -= .5
#                 if self.gain < 0:
#                     self.gain = 0
#         except AttributeError:
#             if key == keyboard.Key.space:  # stop preview
#                 self.stop_preview = True


# if __name__ == '__main__':
#     print('use main.py')
#     #exit(-1)
#     sys.exit()