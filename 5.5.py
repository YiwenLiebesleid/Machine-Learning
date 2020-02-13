import os
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
from keras.utils.np_utils import to_categorical
from keras import Model
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten
from keras.applications import inception_v3
from keras.optimizers import Adam, SGD
from keras.layers.convolutional import AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

batch_size = 20
num_classes = 5
epochs = 30
data_augmentation = False

times = 500 // batch_size

# input path -> image
def get_input(path):
    image = Image.open(path)
    arr = np.array(image)
    arr = arr / 255
    #arr[arr < 128] = 0
    #arr[arr >=128] = 1
    return(arr)

# input path -> label
def get_label(path):
    return np.int64(path.split('/')[-1].split('_')[0])

def image_generator(files, size, scale):  # scale is a (0,1)
    batch_paths = np.random.choice(a=files, size=size, replace=False)  # choose a batchsize of files (path)
    train_num = int(size * scale)

    batch_data = []
    batch_label = []
    rest_data = []
    rest_label = []

    for input_path in batch_paths[0:train_num]:
        data = get_input(input_path)
        label = get_label(input_path)

        batch_data += [data]
        batch_label += [label]

        files.remove(input_path)

    for input_path in batch_paths[train_num:size]:
        data = get_input(input_path)
        label = get_label(input_path)

        rest_data += [data]
        rest_label += [label]

        files.remove(input_path)

    batch_data = np.array(batch_data)
    batch_label = to_categorical(np.array(batch_label), num_classes=5)
    rest_data = np.array(rest_data)
    rest_label = to_categorical(np.array(rest_label), num_classes=5)

    return batch_data, batch_label, rest_data, rest_label, files

def batch_gen(filelist):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    random.shuffle(filelist)
    
    for t in tqdm(range(times)):
        train_batch, train_batch_label, test_batch, test_batch_label, fileleft = image_generator(filelist,batch_size,0.8)
        train_data.append(train_batch)
        train_label.append(train_batch_label)
        test_data.append(test_batch)
        test_label.append(test_batch_label)
        filelist = fileleft

    train_data = np.vstack(train_data)
    train_label = np.vstack(train_label)
    test_data = np.vstack(test_data)
    test_label = np.vstack(test_label)

    return train_data, train_label, test_data, test_label, filelist

def file_gen():
    path = "../input/mldata/data1"
    items = os.listdir(path)
    filelist = []
    for item in items:
        filelist.append(os.path.join(path,item))
    return filelist

def inception_model(train_data, train_label, test_data, test_label):
    input_tensor = Input(shape=(299,299,3))
    base_model = inception_v3.InceptionV3(input_tensor=input_tensor,include_top=False, weights='imagenet')
    x = base_model.output
    x = AveragePooling2D(pool_size=(8,8))(x)
    x = Flatten()(x)
    #x = Dense(10, activation='relu')(x)
    predictions = Dense(5,activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in model.layers[0:-1]:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['acc'])
    #model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),loss='categorical_crossentropy',metrics=['acc'])
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(train_data, train_label, batch_size=batch_size,
                  epochs=epochs, validation_data=(test_data, test_label),
                  shuffle=True, verbose=2)
    else:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=45,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            shear_range=0.,  # set range for random shear
            zoom_range=[0.8, 1.2],  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0
        )

        datagen.fit(train_data)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(train_data, train_label,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(test_data, test_label),steps_per_epoch=50,verbose=2)

    model.save("5.h5")

if __name__=='__main__':
    filelist = file_gen()

    train_data, train_label, test_data, test_label, filelist = batch_gen(filelist)
    inception_model(train_data,train_label,test_data,test_label)
