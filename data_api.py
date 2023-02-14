import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'

# if __name__ == "__main__":
#     X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     Y = np.arange(0, 1000)
#     data_X= tf.data.Dataset.from_tensor_slices(X)
#     data_Y= tf.data.Dataset.from_tensor_slices(Y)
#
#     for instance in data_X:
#         print(instance)
#
#     for instance in data_Y.take(5): # takes the value from dataset starting from top
#         print(instance)
#
#     AUTOTUNE = tf.data.AUTOTUNE
#     data_X = data_X.map(lambda value: value**2, num_parallel_calls= AUTOTUNE)
#     data_Y = data_Y.map(lambda value: value**3, num_parallel_calls= AUTOTUNE)
#
#     for instance in data_Y.take(5):  # takes the value from dataset starting from top
#         print(instance)
#
#     data_Y = data_Y.shuffle(buffer_size=100).batch(32, num_parallel_calls= AUTOTUNE).prefetch(AUTOTUNE)
# # Read about prefetch
#
#     for instance in data_Y.take(1):  # takes the value from dataset starting from top
#         print(instance)
#
#     print(tf.data.Dataset.cardinality(data_Y))


if __name__=="__main__":
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

    AUTOTUNE = tf.data.AUTOTUNE
    BUFFER_SIZE = 1000

    #step 1: Convert into tf datasets
    train = tf.data.Dataset.from_tensor_slices(tensors=(X_train, Y_train))
    test = tf.data.Dataset.from_tensor_slices(tensors=(X_test, Y_test))

    # for img, label in train.take(1):
    #     print(img, label)

    #Step 2: Create a validation set
    train_size = int(tf.data.Dataset.cardinality(train).numpy() *0.80)
    temp = train.shuffle(BUFFER_SIZE)
    train = temp.take(train_size)
    val = temp.skip(train_size)

    print(train.cardinality(), val.cardinality())

    #step3: Normalize the datasets

    def Normalize(image, label):
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)

        return tf.divide(image, 255.), label

    train = train.map(Normalize, num_parallel_calls= AUTOTUNE)
    val = val.map(Normalize, num_parallel_calls= AUTOTUNE)
    test = test.map(Normalize, num_parallel_calls= AUTOTUNE)

    # for img, label in train:
    #     print(np.min(img.numpy()))
    #     print(np.max(img.numpy()))

    train = train.map(lambda img, label: (tf.expand_dims(img, axis=-1), label), num_parallel_calls=AUTOTUNE)
    #Batch dataset
    train = train.cache().shuffle(BUFFER_SIZE).batch(64, num_parallel_calls=AUTOTUNE)
    val = val.cache().shuffle(BUFFER_SIZE).batch(64, num_parallel_calls=AUTOTUNE)
    test = test.cache().shuffle(BUFFER_SIZE).batch(64, num_parallel_calls=AUTOTUNE)

    for img, label in train.take(1):
        print(img.shape)
        print(label.shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=[28, 28]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train, validation_data=val, epochs=10)