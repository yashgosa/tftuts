import os
import tensorflow as tf
import glob
from sklearn.preprocessing import LabelEncoder
def process_files(absolute_path):
    image = []
    label = []

    for instance in absolute_path:
        image.append(instance)
        temp = instance.split('\\')[-2]
        label.append(temp)

    return image, label

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_image(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, size=(256, 256))
    return image, tf.cast(label, tf.float32)



if __name__ == "__main__":

    AUTOTUNE=tf.data.AUTOTUNE
    BUFFERSIZE= 800

    # origin_url = 'https://www.kaggle.com/datasets/imsparsh/flowers-dataset/download?datasetVersionNumber=2'
    # path = tf.keras.utils.get_file('flower_dataset', origin= origin_url, untar=True)
    # step 1: Retrive the path of the images
    root_dir = r'C:\Users\Yash PC\PycharmProjects\tftuts\data\train'

    files = []

    # for instance in os.listdir(root_dir):
    #     for flower in os.listdir((os.path.join(root_dir, instance))):
    #         files.append(os.path.join(root_dir, instance, flower))

    files = glob.glob(os.path.join(root_dir, '*', '*.jpg'), recursive= True)
    print(len(files))



    image_path, label_path = process_files(files)

    print(image_path[:2])
    print(label_path[:2])

    #Label encoding
    encoder = LabelEncoder()
    labels = encoder.fit_transform(label_path)
    print(labels[:5])

    # validation set

    dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
    train_size = int(dataset.cardinality().numpy()*0.8)
    temp = dataset.shuffle(BUFFERSIZE)
    train = temp.take(train_size)
    test = temp.skip(train_size)


    val_size = int(train.cardinality().numpy() * 0.2)
    temp = train.shuffle(BUFFERSIZE)
    val = temp.take(val_size)
    train = temp.skip(val_size)

    # print(train.cardinality(), test.cardinality(),
    #       val.cardinality())

    train = train.map(read_image, num_parallel_calls=AUTOTUNE)
    test = test.map(read_image, num_parallel_calls=AUTOTUNE)
    val = val.map(read_image, num_parallel_calls=AUTOTUNE)

    train = train.cache().shuffle(BUFFERSIZE).batch(32).prefetch(AUTOTUNE)
    test = test.cache().shuffle(BUFFERSIZE).batch(32).prefetch(AUTOTUNE)
    val = val.cache().shuffle(BUFFERSIZE).batch(32).prefetch(AUTOTUNE)


    #
    """'train = train.map(lambda image, label: (tf.divide(image, 255.), label))
    test = test.map(lambda image, label: (tf.divide(image, 255.), label))
    val = val.map(lambda image, label: (tf.divide(image, 255.), label))"""


    # Data Augmentation

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.2),
    ])

    rescale = tf.keras.layers.Rescaling(1/ 255.)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256, 256, 3)),
        rescale,
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5)
    ])

    print(model.summary())






