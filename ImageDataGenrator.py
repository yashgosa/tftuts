import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == '__main__':
    print("Loading...")

    root_dir = r'C:\Users\Yash PC\PycharmProjects\tftuts\data\train'

    data_gentrator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1. /255,
        rotation_range= 5, #Degree of the range of random rotation
        featurewise_center= True, # Sets the input mean to 0
        featurewise_std_normalization=True,
        horizontal_flip=True,
        zoom_range=[0.8, 0.9], #Lower, upper range
        brightness_range=[0.1, 0.4],
        data_format='channels_last', #(x, y, 3)
        validation_split= 0.2,
        dtype= np.float32
    )

    train_generator = data_gentrator.flow_from_directory(
        directory= root_dir,
        target_size= (256, 256),
        class_mode='sparse',
        batch_size= 32,
        shuffle = True,
        subset='training',
        seed= 0
    )

    val_generator = data_gentrator.flow_from_directory(
        directory=root_dir,
        target_size=(256, 256),
        class_mode='sparse',
        batch_size= 32,
        shuffle=True,
        subset='validation',
        seed=0
    )

    #This is a generator not a tf dataset, this generator outputs image infintely

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, activation= 'relu', input_shape=(256, 256, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, activation= 'softmax')
    ])

    model.compile(loss = tf.losses.sparse_categorical_crossentropy,
                  optimizer= 'adam',
                  metrics=['accuracy'])

    #len(training_data) // batch_size <= steps_per_epoch

    history = model.fit(train_generator, validation_data= val_generator, epochs= 5,
                        steps_per_epoch=25)


