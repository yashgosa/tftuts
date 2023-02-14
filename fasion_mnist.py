import os
import graphviz
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    fashion_mnist=tf.keras.datasets.fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

    # print(X_train.shape)
    # print(Y_test.shape)
    # print(X_test.shape)
    # print(Y_train.shape)

    # plt.imshow(X_train[0])
    # plt.show()
    # print(X_train.dtype) #uint8
    X_train = tf.cast(X_train, tf.float32)
    Y_train = tf.cast(Y_train, tf.float32)
    X_test = tf.cast(X_test, tf.float32)
    Y_test = tf.cast(Y_test, tf.float32)

    # Create a validation set: 80 20 rule
    train_size = int(len(X_train) * 0.80)
    index = tf.range(len(X_train))
    # print(index)
    index = tf.random.shuffle(index)
    # print(index)
    train = tf.gather(X_train, index[: train_size])
    train_labels = tf.gather(Y_train, index[: train_size])
    validation = tf.gather(X_train, index[train_size: ])
    validation_labels = tf.gather(Y_train, index[train_size: ])
    #
    # print(train.shape)
    # print(validation.shape)

    # IMPORTANT: When applying gradient descent we should normalize our data. If the gradient fluctutes a lot then it fail to halt at a minima
    train = tf.divide(train, 255.) #uint --> unsigned int
    validation = tf.divide(validation, 255.)
    test = tf.divide(X_test, 255.)

    ###################################### DATA PREPROCESSING PART ######################################

    # In tf we have several ways:
    # 1. sequential mode
    # 2. Functional model api
    # 3. Subclassing model api

    #train = (48000, 28, 28)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=train.shape[1:]),  # np.ravel() does the same,
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# """
#     # tf.keras.utils.plot_model(model, show_dtype=True, show_shapes=True, show_layer_names=True)
#     CHECK WHAT IS THE ISSUE WITH THIS LINE!!!
#
# """
    # input_layer = tf.keras.layers.Input(shape=train.shape[1:])
    # flatten_layer = tf.keras.layers.Flatten()(input_layer)
    # dense_layer = tf.keras.layers.Dense(100, activation = 'relu')(flatten_layer)
    # output_layer = tf.keras.layers.Dense(10, activation='softmax')(dense_layer)
    #
    # model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])

    #
    # for layer in model.layers:
    #     print(layer.name)
    #
    # print(model.layers[2].get_weights())

    # model.get_weights()
    # model_clone = tf.keras.models.clone_model(model)
    # print(model_clone.get_weights())
    # model_clone.set_weights(model)
    # print(model_clone.get_weights())




    # #This remains the same in sequential and in the
    # print(model.summary())
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer='adam', metrics=['accuracy'])
    history = model.fit(train, train_labels, validation_data=(validation, validation_labels), epochs = 10)
    model.save('my_fasion_mnist.h5')

    model = tf.keras.models.load_model('my_fasion_mnist.h5')
    # pd.DataFrame(history.history).plot(figsize=(8, 8))
    # plt.grid()
    # plt.show()
    print(graphviz.__version__)

    pred = model.predict(X_test)
    print(pred.shape)

    pred = np.argmax(pred, axis=1)
    print(np.sum(pred == Y_test)/ len(pred))

    print(pred[0])
