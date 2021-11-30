import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

print("Libraries updated")

import cv2
from glob import glob

print("Libraries updated")

import tensorflow.compat.v1 as tf
#import tensorflow as tf
from tensorflow import keras
#import segmentation_models as sm

print("Libraries updated")

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,MaxPool2D
from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image

from sklearn.model_selection import train_test_split


print("Libraries updated")


img = mpimg.imread('LabelledRice/Healthy/IMG_2996.jpg')
imgplot = plt.imshow(img)
plt.show()

##test_image = image.load_img('LabelledRice/Healthy/IMG_2996.jpg', target_size = (128, 128))
##input1 = cv2.imread('LabelledRice/Healthy/IMG_2996.jpg')
##cv2.imshow('Input - ', input1)
##test_image = image.img_to_array(test_image)
##test_image = np.expand_dims(test_image, axis = 0)
##result = model.predict(test_image)
##print (result[0][0])
##print('Detected: {}'.format(class_names[result]))
##


# Set some standard parameters upfront
pd.options.display.float_format = '{:.2f}'.format
sns.set() # Default seaborn look and feel
plt.style.use('ggplot')
print('pandas version ', pd.__version__ , '\n' )


data_dir = './LabelledRice'
images = glob(os.path.join(data_dir, '*/*.jpg'))
total_images = len(images)
print('Total images:', total_images, '\n' )

# number of images per class
image_count = []
class_names = []

for folder in os.listdir(os.path.join(data_dir)):
    folder_num = len(os.listdir(os.path.join(data_dir, folder)))
    image_count.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)
    

X_Feat = list()
Y_Label = list()
IMG_SIZE = 128
for i in os.listdir("./LabelledRice/LeafBlast"):
    try:
        path = "./labelledrice/LeafBlast/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        X_Feat.append(img)
        Y_Label.append(0)
    except:
        None
print('\n', "Image set 0 - LeafBlast updated")

for i in os.listdir("./labelledrice/BrownSpot"):
    try:
        path = "./labelledrice/BrownSpot/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        X_Feat.append(img)
        Y_Label.append(1)
    except:
        None
print("Image set 1 - BrownSpot updated")
        
for i in os.listdir("./labelledrice/Healthy"):
    try:
        path = "./labelledrice/Healthy/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        X_Feat.append(img)
        Y_Label.append(2)
    except:
        None
print("Image set 2 - Healthy updated")

for i in os.listdir("./labelledrice/Hispa"):
    try:
        path = "./labelledrice/Hispa/"+i
        img = plt.imread(path)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        X_Feat.append(img)
        Y_Label.append(3)
    except:
        None
print("Image set 3 - Hispa updated")


##print(X_Feat.shape)
##print(Y_Label.shape)
###print(X_Feat(5))
##print(Y_Label(5))
        
X_Feat = np.array(X_Feat)
#print(X_Feat(5))

plt.figure(figsize = (16,16))
for i in range(4):
    img = X_Feat[900*i]
    plt.subplot(1,4,i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(class_names[i])

plt.savefig('Sample Images with Label')
#plt.show()


Y_Label = to_categorical(Y_Label, num_classes = 4)

x_train,x_test,y_train,y_test = train_test_split(X_Feat,Y_Label,test_size = 0.15,random_state = 42)
print("X_train shape: " + str(x_train.shape))
#print("Y_train shape: " + y_train.shape)
print("X_test shape: " + str(x_test.shape))
#print("Y_test shape: " + str(y_test.shape))


print( "CNN Model Build")
model = Sequential()
# 1st Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3),padding="Same",activation="relu" , input_shape = (IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# 2nd Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# 3rd Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# 4th Convolutional Layer
model.add(Conv2D(filters=256,kernel_size = (3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# 5th Convolutional Layer
model.add(Conv2D(filters=512,kernel_size = (3,3),padding="Same",activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())
# 1st Fully Connected Layer
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.5))
model.add(BatchNormalization())
# Add output layer
model.add(Dense(4,activation="softmax"))

print("Model Summary")
model.summary() # print summary my model
print("Model Compile")
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001),metrics=['accuracy']) #optimizer='adam'


#model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['acc'])


epoch = 1
batch_size = 10

print( "Start Training", '\n' )
##history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
##                              epochs= epoch,validation_data=(x_val,y_val),
##                              steps_per_epoch=x_train.shape[0] // batch_size
##                              )

model.fit( x_train,y_train,batch_size=batch_size,epochs= epoch )


#model.fit(x=X_train, y=y_train, batch_size=10, epochs=2,
#          validation_data=(X_val, y_val), callbacks=[checkpoint])

print( "Training End", '\n' )


img = mpimg.imread('LabelledRice/Healthy/IMG_2996.jpg')
imgplot = plt.imshow(img)
plt.show()
test_image = image.load_img('LabelledRice/Healthy/IMG_2996.jpg', target_size = (128, 128))
#input1 = cv2.imread('LabelledRice/Healthy/IMG_2996.jpg')
#cv2.imshow('Input - ', input1)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print (result)
#print('Detected: {}'.format(class_names[result]))


##predict('Healthy/IMG_2996.jpg')
##transform = transforms.Compose([transforms.RandomResizedCrop(224),
##                               transforms.ToTensor(),
##                               transforms.Normalize([0.485, 0.456, 0.406],
##                                                    [0.229, 0.224, 0.225])])
##img = Image.open(img_path)
##img = transform(img)
##img = img.unsqueeze(0)
##prediction = model_vgg(img)
##prediction = prediction.cpu().data.numpy().argmax()
##
##img=mpimg.imread(img_path)
##imgplot = plt.imshow(img)
##plt.show()

#print('Detected: {}'.format(class_names[prediction]))



print("Project End")
