#!/usr/bin/env python
# coding: utf-8

# an <validation_dira href="https://colab.research.google.com/github/asukul/DS301-f19/blob/master/CNN_Image_classification_Cats_Dogs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ##### Copyright 2018 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Image classification

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/images/classification"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/images/classification.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This tutorial shows how to classify cats or dogs from images. It builds an image classifier using a `tf.keras.Sequential` model and load data using `tf.keras.preprocessing.image.ImageDataGenerator`. You will get some practical experience and develop intuition for the following concepts:
# 
# * Building _data input pipelines_ using the `tf.keras.preprocessing.image.ImageDataGenerator` class to efficiently work with data on disk to use with the model.
# * _Overfitting_ —How to identify and prevent it.
# * _Data augmentation_ and _dropout_ —Key techniques to fight overfitting in computer vision tasks to incorporate into the data pipeline and image classifier model.
# 
# This tutorial follows a basic machine learning workflow:
# 
# 1. Examine and understand data
# 2. Build an input pipeline
# 3. Build the model
# 4. Train the model
# 5. Test the model
# 6. Improve the model and repeat the process

# ## Import packages

# Let's start by importing the required packages. The `os` package is used to read files and directory structure, NumPy is used to convert python list to numpy array and to perform required matrix operations and `matplotlib.pyplot` to plot the graph and display images in the training and validation data.

# In[2]:


from __future__ import absolute_import, division, print_function, unicode_literals


# Import Tensorflow and the Keras classes needed to construct our model.

# In[3]:


try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf


# In[4]:


# Load the TensorBoard notebook extension
get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


#using multiple GPUs is to use tf.distribute.Strategy
#https://www.tensorflow.org/guide/gpu

tf.debugging.set_log_device_placement(True)

strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
#  inputs = tf.keras.layers.Input(shape=(1,))
#  predictions = tf.keras.layers.Dense(1)(inputs)
#  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
#  model.compile(loss='mse',
#                optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))


# In[7]:


#https://www.tensorflow.org/guide/gpu

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# ## Load data

# Begin by downloading the dataset. This tutorial uses a filtered version of <a href="https://www.kaggle.com/c/dogs-vs-cats/data" target="_blank">Dogs vs Cats</a> dataset from Kaggle. Download the archive version of the dataset and store it in the "/tmp/" directory.

# In[8]:


# _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')


# In[9]:


# The path to our data
#PATH = '/home/ruski/Documents/Research/CyAdsClassifier_DL_data/data/'
#PATH = 'C:\GCloud_bucket\cyads-classifier-data\cyads-classifier-data\CyAdsClassifier_DL_data\data_images'
PATH = 'C:\GCloud_bucket\cyads-classifier-data\cyads-classifier-data\CyAdsClassifier_DL_data\data_images'


# The dataset has the following directory structure:
# 
# <pre>
# <b>cats_and_dogs_filtered</b>
# |__ <b>train</b>
#     |______ <b>cats</b>: [cat.0.jpg, cat.1.jpg, cat.2.jpg ....]
#     |______ <b>dogs</b>: [dog.0.jpg, dog.1.jpg, dog.2.jpg ...]
# |__ <b>validation</b>
#     |______ <b>cats</b>: [cat.2000.jpg, cat.2001.jpg, cat.2002.jpg ....]
#     |______ <b>dogs</b>: [dog.2000.jpg, dog.2001.jpg, dog.2002.jpg ...]
# </pre>

# After extracting its contents, assign variables with the proper file path for the training and validation set.

# In[10]:


train_dir = os.path.join(PATH, 'train')
# validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')


# In[11]:


# train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures

# directory with non political images for training
train_non_political_dir = os.path.join(train_dir, 'non-political')
# train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures

# directory with political images for training
train_political_dir = os.path.join(train_dir, 'political')
# validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures

# directory with non political images to testing
test_non_political_dir = os.path.join(test_dir, 'non-political')
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# directory with political images for testing
test_political_dir= os.path.join(test_dir, 'political')  # directory with our validation dog pictures


# ### Understand the data

# Let's look at how many cats and dogs images are in the training and validation directory:

# # TODO  Make sure this can count all of the files in the subfolders

# In[12]:


num_non_political_tr= len(os.listdir(train_non_political_dir))
num_political_tr = len(os.listdir(train_political_dir))

num_non_political_test = len(os.listdir(test_non_political_dir))
num_political_test= len(os.listdir(test_political_dir))

total_train = num_non_political_tr + num_political_tr
total_test = num_non_political_test + num_political_test


# In[13]:


print('total training non-political images:', num_non_political_tr)
print('total training political images:', num_political_tr)

print('total test non-political images:', num_non_political_test)
print('total test political images:', num_political_test)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_test)


# For convenience, set up variables to use while pre-processing the dataset and training the network.

# In[14]:


batch_size = 128
#batch_size = 16
# batch_size = 64
epochs = 10
# epochs = 30
# epochs = 20

IMG_HEIGHT = 150
IMG_WIDTH = 150
#IMG_HEIGHT = 48
#IMG_WIDTH = 48


# ## Data preparation

# Format the images into appropriately pre-processed floating point tensors before feeding to the network:
# 
# 1. Read images from the disk.
# 2. Decode contents of these images and convert it into proper grid format as per their RGB content.
# 3. Convert them into floating point tensors.
# 4. Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.
# 
# Fortunately, all these tasks can be done with the `ImageDataGenerator` class provided by `tf.keras`. It can read images from disk and preprocess them into proper tensors. It will also set up generators that convert these images into batches of tensors—helpful when training the network.

# In[15]:


# Generator for our training data
train_image_generator = ImageDataGenerator(rescale=1./255)
# Generator for our validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)


# After defining the generators for training and validation images, the `flow_from_directory` method load images from the disk, applies rescaling, and resizes the images into the required dimensions.

# In[16]:


train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# In[17]:


val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')


# ### Visualize training images

# Visualize the training images by extracting a batch of images from the training generator—which is 32 images in this example—then plot five of them with `matplotlib`.

# In[18]:


sample_training_images, _ = next(train_data_gen)


# The `next` function returns a batch from the dataset. The return value of `next` function is in form of `(x_train, y_train)` where x_train is training features and y_train, its labels. Discard the labels to only visualize the training images.

# In[19]:


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[20]:


plotImages(sample_training_images[:5])


# ## Create the model

# The model consists of three convolution blocks with a max pool layer in each of them. There's a fully connected layer with 512 units on top of it that is activated by a `relu` activation function. The model outputs class probabilities based on binary classification by the `sigmoid` activation function.

# In[21]:


model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


# ### Compile the model
# 
# For this tutorial, choose the *ADAM* optimizer and *binary cross entropy* loss function. To view training and validation accuracy for each training epoch, pass the `metrics` argument.

# In[22]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# ### Model summary
# 
# View all the layers of the network using the model's `summary` method:

# In[23]:


model.summary()


# ### Train the model

# Use the `fit_generator` method of the `ImageDataGenerator` class to train the network.

# In[24]:


#add tensorboard
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[ ]:


#history = model.fit_generator(  <update to .fit for TF 2.2.0
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_test // batch_size
)


# In[ ]:


# history = model.fit(
#     train_data_gen,
#     steps_per_epoch=total_train // batch_size,
#     epochs=epochs,
#     validation_data=val_data_gen,
#     validation_steps=total_test // batch_size
# )


# ### Visualize training results

# Now visualize the results after training the network.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# As you can see from the plots, training accuracy and validation accuracy are off by large margin and the model has achieved only around **70%** accuracy on the validation set.
# 
# Let's look at what went wrong and try to increase overall performance of the model.

# ## Overfitting

# In the plots above, the training accuracy is increasing linearly over time, whereas validation accuracy stalls around 70% in the training process. Also, the difference in accuracy between training and validation accuracy is noticeable—a sign of *overfitting*.
# 
# When there are a small number of training examples, the model sometimes learns from noises or unwanted details from training examples—to an extent that it negatively impacts the performance of the model on new examples. This phenomenon is known as overfitting. It means that the model will have a difficult time generalizing on a new dataset.
# 
# There are multiple ways to fight overfitting in the training process. In this tutorial, you'll use *data augmentation* and add *dropout* to our model.

# ## Data augmentation

# Overfitting generally occurs when there are a small number of training examples. One way to fix this problem is to augment the dataset so that it has a sufficient number of training examples. Data augmentation takes the approach of generating more training data from existing training samples by augmenting the samples using random transformations that yield believable-looking images. The goal is the model will never see the exact same picture twice during training. This helps expose the model to more aspects of the data and generalize better.
# 
# Implement this in `tf.keras` using the `ImageDataGenerator` class. Pass  different transformations to the dataset and it will take care of applying it during the training process.

# ### Augment and visualize data

# Begin by applying random horizontal flip augmentation to the dataset and see how individual images look like after the transformation.

# ### Apply horizontal flip

# Pass `horizontal_flip` as an argument to the `ImageDataGenerator` class and set it to `True` to apply this augmentation.

# In[22]:


image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)


# In[23]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))


# Take one sample image from the training examples and repeat it five times so that the augmentation is applied to the same image five times.

# In[24]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[25]:


# Re-use the same custom plotting function defined and used
# above to visualize the training images
plotImages(augmented_images)


# ### Randomly rotate the image

# Let's take a look at a different augmentation called rotation and apply 45 degrees of rotation randomly to the training examples.

# In[26]:


image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)


# In[27]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[28]:


plotImages(augmented_images)


# ### Apply zoom augmentation

# Apply a zoom augmentation to the dataset to zoom images up to 50% randomly.

# In[29]:


# zoom_range from 0 - 1 where 1 = 100%.
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5) # 


# In[30]:


train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH))

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


# In[31]:


plotImages(augmented_images)


# ### Put it all together

# Apply all the previous augmentations. Here, you applied rescale, 45 degree rotation, width shift, height shift, horizontal flip and zoom augmentation to the training images.

# In[32]:


image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )


# In[33]:


train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')


# Visualize how a single image would look five different times when passing these augmentations randomly to the dataset.

# In[34]:


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)


# ### Create validation data generator

# Generally, only apply data augmentation to the training examples. In this case, only rescale the validation images and convert them into batches using `ImageDataGenerator`.

# In[35]:


image_gen_val = ImageDataGenerator(rescale=1./255)


# In[36]:


val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 # directory=validation_dir,
                                                 directory=test_dir,
                                                 target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                 class_mode='binary')


# ## Dropout

# Another technique to reduce overfitting is to introduce *dropout* to the network. It is a form of *regularization* that forces the weights in the network to take only small values, which makes the distribution of weight values more regular and the network can reduce overfitting on small training examples. Dropout is one of the regularization technique used in this tutorial
# 
# When you apply dropout to a layer it randomly drops out (set to zero) number of output units from the applied layer during the training process. Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc. This means dropping out 10%, 20% or 40% of the output units randomly from the applied layer.
# 
# When appling 0.1 dropout to a certain layer, it randomly kills 10% of the output units in each training epoch.
# 
# Create a network architecture with this new dropout feature and apply it to different convolutions and fully-connected layers.

# ## Creating a new network with Dropouts

# Here, you apply dropout to first and last max pool layers. Applying dropout will randomly set 20% of the neurons to zero during each training epoch. This helps to avoid overfitting on the training dataset.

# In[37]:


model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', 
           input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])


# ### Compile the model

# After introducing dropouts to the network, compile the model and view the layers summary.

# In[38]:


model_new.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_new.summary()


# # Model checkpoint
# 
# #model Checkpoint
# #https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint

# In[52]:


filepath = "./"

tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch', **kwargs
)


# In[53]:



EPOCHS = 10
checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)

# Model weights are saved at the end of every epoch, if it's the best seen
# so far.
#model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])
#model_new.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

# The model weights (that are considered the best) are loaded into the model.
#model.load_weights(checkpoint_filepath)
#model_new.load_weights(checkpoint_filepath)


# ### Train the model

# After successfully introducing data augmentations to the training examples and adding dropouts to the network, train this new network:

# In[59]:


tf.keras.callbacks.ModelCheckpoint(
    filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch'
)


# In[ ]:


#checkpoint = ModelCheckpoint("save_model_epoch.hdf5", monitor='loss', verbose=1,
#    save_best_only=False, mode='auto', period=1)

#history = model_new.fit_generator(
history = model_new.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    # validation_steps=total_val // batch_size
    validation_steps=total_test// batch_size,
    callbacks=[model_checkpoint_callback]
)


# ### Visualize the model

# Visualize the new model after training, you can see that there is significantly less overfitting than before. The accuracy should go up after training the model for more epochs.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[ ]:


#add tensorboard disply


# In[ ]:


from tensorboard import notebook
notebook.list() # View open TensorBoard instances


# In[ ]:


# Control TensorBoard display. If no port is provided, 
# the most recently launched TensorBoard is used
notebook.display(port=6006, height=1000) 


# In[ ]:


#save the model
model.save("CNN_Image_classification_Ads_TensorBoard.h5")


# In[ ]:




