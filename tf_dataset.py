import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if __name__ == "__main__":

    root_dir = r'C:\Users\Yash PC\PycharmProjects\tftuts\data\train'

    train = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        seed=1234,
        validation_split=0.2,
        subset='training'
    )

    val = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        seed=1234,
        validation_split=0.2,
        subset='validation'
    )

    for img, label in train.take(1):
        print(img.shape, label.shape)