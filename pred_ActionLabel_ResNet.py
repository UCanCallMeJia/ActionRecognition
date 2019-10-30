from keras.models import load_model
from utils import get_frames
from train_ResNet50 import get_data
import sys
from cv2 import cv2
import numpy as np 
import os
from PIL import Image

# 加载训练好的残差网络
def get_resnet50_model():
    resnet50_model = load_model('/home/ubuntu/python_ws/ActionLabel/model/resnet50_motion_primitive.h5')
    return resnet50_model

# 将标签打在视频画面上
def add_ActionLabel_to_frames(labels,frames):
    video = []
    for label,frame in zip(labels,frames):
        # print('Wirte Label to Video :',label)
        font=cv2.FONT_HERSHEY_SIMPLEX#使用默认字体
        img=cv2.putText(frame,str(label),(50,50),font,1.2,(255,255,255),2)#添加文字，1.2表示字体大小，（0,40）是初始的位置，(255,255,255)表示颜色，2表示粗细
        video.append(img)
    return np.array(video,dtype=np.uint8)

# 将数组储存为视频
def frames_to_video(frames,video_name):
    fps = 20
    size = (frames[0].shape[1],frames[0].shape[0])

    
    save_path = os.path.join('/home/ubuntu/python_ws/ActionLabel/videos_add_ActionLabel',video_name.split('.')[0] + '.avi')

    video_writer = cv2.VideoWriter(save_path,
                                    cv2.VideoWriter_fourcc('D','I','V','X'),
                                    fps,
                                    size )
    for img in frames:
        video_writer.write(img)
    video_writer.release()

# 单帧+前后四帧 motion image 输入，使用新的模型推测视频的action label
def predict_ActionLabel_to_NewVideo_new(video_path):
    frames_before_resize = get_frames(video_path,divide=1,show=False)

    frames = []
    for i in range(len(frames_before_resize)):

        image = cv2.resize(frames_before_resize[i],(960,540))
        # print(image.shape)
        frames.append(image) 
    
    frames = np.array(frames)/255.0
    


    x_test = []
    for i in range(2,len(frames)-3):
        motion_1 = frames[i-1] - frames[i]
        motion_2 = frames[i+1] - frames[i]
        motion_3 = frames[i-2] - frames[i]
        motion_4 = frames[i+2] - frames[i]
        x_test.append(np.concatenate((frames[i],motion_1,motion_2,motion_3,motion_4),axis=2))

    x_test = np.array(x_test)


    print('input shape:',x_test.shape)
    print('frames shape:',frames.shape)
    model = get_resnet50_model()
    results = np.array(model.predict(x_test)) 
    # print(results)
    labels = []
    action = {0:'idle',1:'pick',2:'push'}
    for result in results:
        index = result.argmax()
        labels.append(action[index])
    extract_actions(labels)

    video_name = video_path.split('/')[-1]
    video = add_ActionLabel_to_frames(labels,frames_before_resize[2:-3])
    frames_to_video(video,video_name)

# 单帧输入时，使用模型推测视频的action label
def predict_ActionLabel_to_NewVideo(video_path):
    frames_before_resize = get_frames(video_path,show=False)
    frames = []
    for i in range(len(frames_before_resize)):

        image = cv2.resize(frames_before_resize[i],(960,540))
        # print(image.shape)
        frames.append(image) 
    
    frames = np.array(frames)
    print(frames.shape )
    model = get_resnet50_model()
    results = np.array(model.predict(frames/255.0)) 
    # print(results)
    labels = []
    action = {0:'idle',1:'pick',2:'push'}
    for result in results:
        index = result.argmax()
        labels.append(action[index])
    extract_actions(labels)

    video_name = video_path.split('/')[-1]
    video = add_ActionLabel_to_frames(labels,frames)
    frames_to_video(video,video_name)

def extract_actions(labels):
    actions = ['pick','push']
    pred_actions = []
    for label in labels:
        if label in actions:
            pred_actions.append(label)
            actions.remove(label)
    print('extract actions for VREP: ',pred_actions)

def main():
    VIDEO_DIR = '/home/ubuntu/python_ws/ActionLabel/videos/test'
    videonames = os.listdir(VIDEO_DIR)
    for videoname in videonames:
        video_path = os.path.join(VIDEO_DIR,videoname)
        predict_ActionLabel_to_NewVideo_new(video_path)
        # input()

if __name__ == "__main__":
    main()