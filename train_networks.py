from keras.models import Sequential,load_model,save_model,Model
from keras.layers import Flatten,Dense,Input,Conv2D,MaxPooling2D,Dropout,GlobalAveragePooling2D
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet_v2 import  MobileNetV2
from keras.preprocessing import image
from keras.optimizers import adam
from keras.callbacks import TensorBoard
import pickle
import numpy as np 

# GPU config
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

# use MobileNetV2 pretrained model
def my_mobnet():
    net = MobileNetV2(include_top=False,input_shape=(544, 960, 3))
    # resnet.summary()
    x = net.output
    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    x = Dense(32, activation='relu', name='fc3')(x)
    out = Dense(3, activation='softmax', name='predictions')(x)

    model = Model(inputs=net.input, outputs=out)

    model.summary()
    return model

# ResNet50 pretrained model
def my_resnet50():
    Inp = Input((540,960,15))
    a = Conv2D(3, (3, 3), strides=(1, 1),padding='same',activation='relu', input_shape=(544, 960, 15))(Inp)

    resnet = ResNet50(weights='imagenet',include_top=False,input_shape=(540, 960, 3))
    # resnet.summary()
    x = resnet(a)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    # x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', name='fc3')(x)
    out = Dense(3, activation='softmax', name='predictions')(x)

    model = Model(inputs=Inp, outputs=out)

    for layer in resnet.layers:
        layer.trainable = False

    model.summary()
    return model


# vgg16 pretrained model
def my_vgg16():
    # 读取 VGG16-Block 5 舍弃全连接
    vgg_model = VGG16(weights='imagenet', include_top=False,input_shape=(544, 960, 3))
    vgg_model.summary()
    
    # vgg16的卷积层后添加全连接层
    x = vgg_model.output
    x = Flatten(name='flatten')(x)
    x = Dense(12, activation='relu', name='fc1')(x)
    x = Dense(6, activation='relu', name='fc2')(x)
    # x = Dense(4, activation='relu', name='fc3')(x)
    out = Dense(3, activation='softmax', name='predictions')(x)

    # 完整vgg16分类模型
    my_model = Model(input=vgg_model.input, outputs=out)
    # 下面的模型输出中，vgg16的层和参数不会显示出，但是这些参数在训练的时候会更改
    print('\nmy vgg16 model for the task')
    my_model.summary()

    return my_model

# ordinary model test
def test_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(544, 960, 3)))
    # model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(3, activation='softmax'))

    return model

# read training/testing data
def get_data(mode='train'):
    if mode == 'train':
        x_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/train/x_data_train.pkl'
        y_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/train/y_data_train.pkl'
        with open(x_path,'rb') as f:
            x_data = pickle.load(f)
            # normalization the input
            x_data = np.array(x_data)/255.0
            # print(x_data.shape)
        with open(y_path,'rb') as f:
            y_data = pickle.load(f)
            y_data = np.array(y_data)
            # print(y_data.shape)
        return x_data , y_data
    elif mode=='eval':
        x_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/test/x_data_test.pkl'
        y_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/test/y_data_test.pkl'
        with open(x_path,'rb') as f:
            x_data = pickle.load(f)
            # normalization the input
            x_data = np.array(x_data)/255.0
            # print(x_data.shape)
        with open(y_path,'rb') as f:
            y_data = pickle.load(f)
            y_data = np.array(y_data)
            # print(y_data.shape)
        return x_data , y_data

# process per frame into 5 frames(containing time info)
def process_data(x,y):
    x_train = []
    y_train = []
    for i in range(2,len(x)-3):
        motion_1 = x[i-1] - x[i]
        motion_2 = x[i+1] - x[i]
        motion_3 = x[i-2] - x[i]
        motion_4 = x[i+2] - x[i]
        x_train.append(np.concatenate((x[i],motion_1,motion_2,motion_3,motion_4),axis=2))
        y_train.append(y[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # print(x_train.shape,y_train.shape)
    return x_train,y_train

def main(mode='train'):

    if mode=='train':
        x_train,y_train = get_data(mode='train')
        x_train,y_train = process_data(x=x_train, y=y_train)
        # x_train = preprocess_input(x_train)
        y_train = np_utils.to_categorical(y_train,3)

        print(x_train.shape , y_train.shape)

        model = my_resnet50()
        model.summary()
        tbCallBack = TensorBoard(log_dir='./logs')
        ad = adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                       optimizer=ad, metrics=['accuracy'])
        # train on batch
        # for i in range(EPOCHS):
        #     for j in range(int(len(x_train)/BATCH_SIZE)):
        #         x_batch = x_train[j*BATCH_SIZE : (j+1)*BATCH_SIZE]
        #         y_batch = y_train[j*BATCH_SIZE : (j+1)*BATCH_SIZE]
        #         print(x_batch.shape)
        #         loss = model.train_on_batch(x_batch,y_batch)
        #         print(loss)
        model.fit(x_train,y_train,batch_size=2,epochs=200,callbacks=[tbCallBack])
        model.save('/home/ubuntu/python_ws/ActionLabel/model/resnet50_motion_primitive.h5')

        input('Train Done!!')
        # x_eval , y_eval = get_data(mode='eval')
        # y_eval = np_utils.to_categorical(y_eval,5)
        # score = model.evaluate(x_eval,y_eval)
        # print(score)

    elif mode=='eval':
        x_eval , y_eval = get_data(mode='eval')
        y_eval = np_utils.to_categorical(y_eval,5)
        trained_model = load_model('/home/ubuntu/python_ws/ActionLabel/model/resnet50_fruits.h5')
        print(x_eval.shape, y_eval.shape)
        result = trained_model.evaluate(x_eval,y_eval)
        print(result)

if __name__ == "__main__":
    main(mode='train')