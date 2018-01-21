
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
# get_ipython().magic('matplotlib inline')

import keras # broken for keras >= 2.0, use 1.2.2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from utils import load_weights, Box, yolo_net_out_to_car_boxes, draw_box


# ## construct the tiny-yolo model ##

# In[2]:


keras.backend.set_image_dim_ordering('th')


# In[3]:


model = Sequential()
model.add(Convolution2D(16, 3, 3,input_shape=(3,448,448),border_mode='same',subsample=(1,1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(64,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(128,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(256,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(512,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024,3,3 ,border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))


# In[4]:


model.summary()


# ## load weight from pretrained weights for yolo ##
# The weight file can be downloaded from https://pjreddie.com/darknet/yolo/

# In[5]:


load_weights(model,'./yolo-tiny.weights')


# ## apply the model to a test image ##

# visualize the box on the original image

# more examples

# In[6]:


images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
batch = np.array([np.transpose(cv2.resize(image[300:650,500:,:],(448,448)),(2,0,1)) 
                  for image in images])
batch = 2*(batch/255.) - 1
out = model.predict(batch)
f,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(11,10))
# boxes = []
for i,ax in zip(range(len(batch)),[ax1,ax2,ax3,ax4,ax5,ax6]):
    boxes = yolo_net_out_to_car_boxes(out[i], threshold = 0.17)
    count =0
#     for b in boxes:
#         count+=1
#         print(count)
#         print(b.x)
#         print(b.y)
#         print(b.h)
#         print(b.w)
    ax.imshow(draw_box(boxes,images[i],[[500,1280],[300,650]]))


# ## apply to video ##

# In[7]:


# Previousboxes = []
def frame_func(image):
#     global Previousboxes
    crop = image[300:650,500:,:]
    resized = cv2.resize(crop,(448,448))
    batch = np.array([resized[:,:,0],resized[:,:,1],resized[:,:,2]])
    batch = 2*(batch/255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)
    boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
    Previousboxes = boxes
    return draw_box(boxes,image,[[500,1280],[300,650]])


# In[8]:


project_video_output = './project_video_output.mp4'
clip1 = VideoFileClip("LaneDetection_output_videos/laneDetection.mp4")


# In[ ]:


boxes = []
lane_clip = clip1.fl_image(frame_func) #NOTE: this function expects color images!!
get_ipython().magic('time lane_clip.write_videofile(project_video_output, audio=False)')

