import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rescaling(img, label):
    return tf.divide(tf.cast(img, tf.float32), 255.), label

"""
x = [[2, 2],
    [3, 3]]
    
tf.tiles(x, multiples = (2, 2)

    |x           x|
    |             |
    |             |
    |x           x|
    ---------------
"""
def augmentation(img, label):
    img = tf.image.random_brightness(img, max_delta= 0.1)

    #RGP image ----> [x, y, 3];  [x, y, 1] after repeating image for 3 times we get [x, y, 3]
    if tf.random.uniform(shape=(), minval=0, maxval=1) <= 0.2:
        grayscale = tf.image.rgb_to_grayscale(img)
        img = tf.tile(grayscale, multiples=[1, 1, 1, 3])

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_contrast(img, lower = 0.2, upper= 0.4)

    return img, label

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

    # for img, label in train.take(1):
    #     print(img.shape, label.shape)

    AUTOTUNE = tf.data.AUTOTUNE
    BUFFER_SIZE = 1000

    train = train.map(rescaling,num_parallel_calls=AUTOTUNE)
    val = val.map(rescaling,num_parallel_calls=AUTOTUNE)

    train = train.map(augmentation, num_parallel_calls=AUTOTUNE)
    val = val.map(augmentation, num_parallel_calls=AUTOTUNE)

    train = train.cache().shuffle(BUFFER_SIZE).prefetch(AUTOTUNE)
    val = val.cache().shuffle(BUFFER_SIZE).prefetch(AUTOTUNE)

    for img, label in train.take(1):
        print(img.shape, label.shape)
