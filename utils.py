import numpy as np
import os
from PIL import Image
import csv
import matplotlib.pyplot as plt
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

# Videos to frames
def get_frames(video_path, divide=1, show=True):
    video = cv2.VideoCapture(video_path)
    frame_num = 0
    frames = []
    if video.isOpened() == False:
        print('unable to open video')
        exit()
    while(True):
        success, frame = video.read()
        if not success :break
        # 提取视频帧数的比率，1 = 100%   2 = 50%
        if frame_num%divide == 0:
            if show:
                cv2.imshow('window',frame)
                cv2.waitKey(10)
            # name = '{}.jpg'.format(frame_num)
            # print('current frame num is: ',frame_num)

            frames.append(frame)
        frame_num += 1
    if show:
        cv2.destroyAllWindows()
    video.release()
    print('read video [{}] successful.'.format(video_path),'Frame number:',len(frames))
    frames = np.array(frames)
    return frames


# process video , generate dataset , generate csv files
def add_label_and_save_dataset(video_path, divide=1):
    video = cv2.VideoCapture(video_path)
    frame_num = 0
    frames = []
    if video.isOpened() == False:
        print('unable to open video :{}'.format(video_path))
        exit()
    while(True):
        success, frame = video.read()
        if not success :break
        # 提取视频帧数的比率，1 = 100%   2 = 50%
        if frame_num%divide == 0:
            cv2.imshow('window',frame)
            cv2.waitKey(10)
            name = '{}.jpg'.format(frame_num)
            print(name)

            frames.append(frame)
        frame_num += 1
    if show:
        cv2.destroyAllWindows()
    video.release()
    print('read video successful. Frame Numbers: ',len(frames))
    frames = np.array(frames)
    return frames

# get the file names with correct num sequence
def filenames(dir):
    names = os.listdir(dir)
    names = [int(num.split('.')[0]) for num in names]
    names.sort()
    file_names = ['{}.jpg'.format(num) for num in names]
    print(file_names)

# visualize the video that added labels
def visualize(sequence_path):
    with open(sequence_path,'r') as f:
        csv_file = csv.reader(f)
        hand_x = []
        hand_y = []
        mouse_x = []
        mouse_y = []
        for i in csv_file:
            # print(i[2])
            hand_x.append(i[2])
            hand_y.append(i[3])
            mouse_x.append(i[4])
            mouse_y.append(i[5])
        hand_x = np.array(hand_x).astype(np.float)
        hand_y = np.array(hand_y).astype(np.float)
        mouse_x = np.array(mouse_x).astype(np.float)
        mouse_y = np.array(mouse_y).astype(np.float)
        print(hand_x.shape,hand_y.shape)
        plt.rcParams['figure.figsize'] = (5,9)
        plt.plot(hand_x,hand_y,'g',mouse_x,mouse_y,'r')
        # plt.scatter(mouse_x,mouse_y,c='b')
        plt.show()


if __name__ == "__main__":
    pass
    # frames = get_frame(r'')
    # filenames(r'')
    # visualize(r'')
    