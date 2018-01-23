import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import math
import random
import os
import argparse
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
import time
import tensorflow.contrib.slim as slim
#import slimmodels.research.slim.nets.nasnet.nasnet
from tensorflow.contrib.keras import optimizers, layers, models, callbacks, utils, preprocessing

random.seed(os.urandom(9))

# tensorboard --logdir=train:./logs/train,test:./logs/test/
# python train.py --dataset ./MNIST-Normal/ --dimensions 3 --augmentation True --tensorboard True

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='./Caltech101-Big2/')  # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--max_batch_size", help="batch_size", default=16)
parser.add_argument("--epochs", help="Number of epochs to train", default=200)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
args = parser.parse_args()

# Hyperparameter
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
height = int(args.height)
channels = int(args.dimensions)
classes = os.listdir(args.dataset + 'TRAIN/')
n_classes = sum((os.path.isdir(args.dataset + 'TRAIN/' + i) for i in classes))
print(str(n_classes) + ' Classes to train')

train_data_dir = './Canny_Caltech101/TRAIN'
validation_data_dir = './Event_Caltech101/TRAIN'
nb_train_samples=32968
nb_validation_samples=6627*2

model=tf.keras.applications.mobilenet.MobileNet(alpha=1.2,input_tensor=tf.keras.layers.Input(shape=(height, width, channels)), classes=n_classes, weights=None)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged


def preproces(x):
	x = x / 255.0 - 0.5
	return x



data_gen_args = dict(preprocessing_function=preproces,
					rotation_range=10,
					width_shift_range=0.20,
					height_shift_range=0.20,
					shear_range=1,
					zoom_range=0.15,
					horizontal_flip=True,
					vertical_flip=False)




# this is the augmentation configuration we will use for training
train_datagen = preprocessing.image.ImageDataGenerator(**data_gen_args)
# Other options:rotation_range, height_shift_range, featurewise_center, vertical_flip, featurewise_std_normalization...
# Also you can give a function as an argument to apply to every iamge


# this is the augmentation configuration we will use for testing:
test_datagen = preprocessing.image.ImageDataGenerator(preprocessing_function=preproces)

# Generator of images from the data folder
train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(height, width),
													batch_size=max_batch_size, class_mode='categorical', shuffle=True)

validation_generator = test_datagen.flow_from_directory(validation_data_dir, target_size=(height, width),
														batch_size=(max_batch_size), class_mode='categorical', shuffle=True)
model.summary() 

adam = optimizers.Adam(0.0001) # decay=0.0001? decay 1/(1+decay*epochs*batches_per_epoch)*lr
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


'''
filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
'''
model.fit_generator(train_generator, steps_per_epoch=nb_train_samples // max_batch_size, epochs=total_epochs, shuffle=True, 
		validation_data=validation_generator, validation_steps=nb_validation_samples // max_batch_size)



score = model.evaluate_generator(validation_generator, nb_validation_samples)
print(score)

