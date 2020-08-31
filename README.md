# ActionRecognition
Action recognition in videos with pretrained Resnet50.

# Introduction 
  
**Platform** Python3.6  
**Libraries** `Numpy`, `Tensorflow`, `Keras`, `opencv-python`, `matplotlib`  
**Hardware** Camera,PC

# Usage  
**Notice** :*The dataset is too big to up load. So firstly you have to make your own dataset.*  
### **Generate your video dataset :**  
Firstly, record some action videos with your camera. Then, run **video_manully_add_label.py** to generate pickle dataset.    
 ```python
 # change the path
x_data_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/train/x_data_train.pkl'
with open(x_data_path,'wb') as f:
    pickle.dump(x_data,f,0)

 # save the label for training data
y_data_path = '/home/ubuntu/python_ws/ActionLabel/generated_dataset/train/y_data_train.pkl'
with open(y_data_path,'wb') as f:
    pickle.dump(y_data,f,0)
 ```  
 Make sure that the video path and save path are corresponding to yours.  

### **Train Action Recognition Network :**  

Run **train_networks.py** to train the model.  
Some pretrained models are used like Resnet50,VGG16 and so on. The models are pretrained on 'imagenet'. You can choose one you like to extract features.  
```python
    if mode=='train':
        x_train,y_train = get_data(mode = 'train')
        x_train,y_train = process_data(x=x_train, y=y_train)
        y_train = np_utils.to_categorical(y_train,3)
        # choose your model here.
        model = my_resnet50()
        model.summary()
```

### **Model evaluation and visualization :**  
In *pred_ActionLabel_ResNet.py*, you can load trained model to evaluate on testset.  
Functions `predict_ActionLabel_to_NewVideo_new()`, `add_ActionLabel_to_frames()`, `frames_to_videos()` are used to visualize the predict label on videos. Follow instructions in the code.

Contact： jiazx@buaa.edu.cn
