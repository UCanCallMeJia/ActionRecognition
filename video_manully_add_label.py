'''
This code is used to add labels to video frames.
Usage: open a video, manully add labels frame by frame.
Number of labels can be changed in the code. 
'''

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import os
import csv
import pickle
import numpy as np
from cv2 import cv2 

def main():
    # change this path corresponding to your sys
    TRAIN_VIDEO_DIR = '/home/ubuntu/python_ws/ActionLabel/videos/train'
    # divided frames in video
    DIVIDE = 2

    names = os.listdir(TRAIN_VIDEO_DIR)
    # x_data: frames  |   y_data: labels 
    x_data = []
    y_data = []
    for name in names:
        # in order to check the dataset, save labeling results
        csv_path = '/home/ubuntu/python_ws/ActionLabel/generated_csvfile/{}.csv'.format(name.split('.')[0])
        print(csv_path)
        with open(csv_path,'w',newline='') as f:
            csv_writer = csv.writer(f)
            video = cv2.VideoCapture(os.path.join(TRAIN_VIDEO_DIR,name))
            frame_num = 0
            frames = []
            if video.isOpened() == False:
                print('unable to open video :{}'.format(name))
                exit()
            while(True):
                label = -1
                success, frame = video.read()
                if not success :break
                # divided ratio，1 = 100%   2 = 50%
                if frame_num%DIVIDE == 0:
                    frame = cv2.resize(frame,(960,540))
                    cv2.imshow('window',frame)
                    cv2.waitKey(10)

                    label = input('label: ')
                    if len(label) == 1:
                        label = int(label)
                    # label = int(input('label: '))
                    label_range = [0,1,2]
                    # add action labels to frames 
                    while True:
                        if label in label_range:
                            break
                        else: 
                            label = input('Wrong Label! give another: ')
                            if len(label) == 1:
                                label = int(label)
                    # label: idle
                    if label == 0:
                        x_data.append(frame)
                        y_data.append(label)
                        write_data = [frame_num,'idle']
                        csv_writer.writerow(write_data)
                    # label: move hand 
                    elif label == 1:
                        x_data.append(frame)
                        y_data.append(label)
                        write_data = [frame_num,'pick']
                        csv_writer.writerow(write_data)
                    # label:  move_apple_to_bowl
                    elif label == 2:
                        x_data.append(frame)
                        y_data.append(label)
                        write_data = [frame_num,'push']
                        csv_writer.writerow(write_data)
                    # # move_banana_to_bowl
                    # elif label == 3:
                    #     x_data.append(frame)
                    #     y_data.append(label)
                    #     write_data = [frame_num,'move_banana_to_bowl']
                    #     csv_writer.writerow(write_data)
                    # # move_kiwi_to_bowl
                    # elif label == 4:
                    #     x_data.append(frame)
                    #     y_data.append(label)
                    #     write_data = [frame_num,'move_kiwi_to_bowl']
                    #     csv_writer.writerow(write_data)
                    frames.append(frame)
                frame_num += 1

            cv2.destroyAllWindows()
            video.release()
            print('read video successful. Frame Numbers: ',len(frames))
            frames = np.array(frames)

    # save the training data
    # change the path
    x_data_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/train/x_data_train.pkl'
    with open(x_data_path,'wb') as f:
        pickle.dump(x_data,f,0)
    # save the label for training data
    y_data_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/train/y_data_train.pkl'
    with open(y_data_path,'wb') as f:
        pickle.dump(y_data,f,0)
        
if __name__ == "__main__":
    main()


