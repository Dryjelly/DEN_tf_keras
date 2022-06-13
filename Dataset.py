import tensorflow as tf
import numpy as np
import gzip

def load_mnist(add_channel = False):
    # Download the mnist dataset using keras
    data_train, data_test = tf.keras.datasets.mnist.load_data()

    # Parse images and labels
    (images_train, labels_train) = data_train
    images_train = images_train.reshape([-1, 28* 28])/255
    if add_channel: images_train = np.tile(images_train, 3)
    #labels_train = tf.keras.utils.to_categorical(labels_train,10)

    (images_test, labels_test) = data_test
    images_test = images_test.reshape([-1, 28* 28])/255
    if add_channel: images_test = np.tile(images_test, 3)
    #labels_test = tf.keras.utils.to_categorical(labels_test,10)
    
    return (images_train, labels_train),(images_test, labels_test)

def load_fashion_mnist(add_channel = False):
    data_train, data_test = tf.keras.datasets.fashion_mnist.load_data()

    # Parse images and labels
    (images_train, labels_train) = data_train
    images_train = images_train.reshape([-1, 28* 28])/255
    if add_channel: images_train = np.tile(images_train, 3)
    #labels_train = tf.keras.utils.to_categorical(labels_train,10)

    (images_test, labels_test) = data_test
    images_test = images_test.reshape([-1, 28* 28])/255
    if add_channel: images_test = np.tile(images_test, 3)
    #labels_test = tf.keras.utils.to_categorical(labels_test,10)
    
    return (images_train, labels_train),(images_test, labels_test)

def load_cifar10():
    data_train, data_test = tf.keras.datasets.cifar10.load_data()

    # Parse images and labels
    (images_train, labels_train) = data_train
    images_train = images_train/255
    labels_train = np.squeeze(labels_train)

    (images_test, labels_test) = data_test
    images_test = images_test/255
    labels_test = np.squeeze(labels_test)
    
    return (images_train, labels_train),(images_test, labels_test)

def read_emnist(data_type = 'train', add_channel = False):
    DATA_SET_DIR = '/home/esdl/tensorflow/DATA_SET/EMNIST_data/'
    
    data_path = DATA_SET_DIR + 'emnist-byclass-{}-images-idx3-ubyte.gz'.format(data_type)
    label_path = DATA_SET_DIR + 'emnist-byclass-{}-labels-idx1-ubyte.gz'.format(data_type)
    
    f_img = gzip.open(data_path,'r')
    f_lab = gzip.open(label_path,'r')

    # non-image information
    buf = f_img.read(16) 
    _ = f_lab.read(8)
    num_images = int.from_bytes(buf[4:8], "big")
    image_size = int.from_bytes(buf[8:12], "big")

    buf = f_img.read()
    images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = images.reshape(-1, image_size, image_size, 1)/255
    images = np.transpose(images,(0,2,1,3)) # inverted horizontally and rotated 90 anti-clockwise
    images = images.reshape(-1, image_size* image_size)
    if add_channel: images = np.tile(images, 3)

    buf = f_lab.read()
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    
    f_img.close()
    f_lab.close()    

    return images, labels

def pick_emnist_class(data_type = 'train', class_type = 'digit', add_channel = False):
    
    images, labels = read_emnist(data_type, add_channel)
    margin = 0 # Set label value range 0 ~ N
    
    if class_type == 'digit':
        pick = [i for i in range(len(labels)) if labels[i] in list(range(0,10))]
    elif class_type == 'upper':
        pick = [i for i in range(len(labels)) if labels[i] in list(range(10,36))]
        margin = 10
    elif class_type == 'lower':
        pick = [i for i in range(len(labels)) if labels[i] in list(range(36,62))]
        margin = 36
    elif class_type == 'english':
        pick = [i for i in range(len(labels)) if labels[i] in list(range(10,62))]
        margin = 10
    else: print("Error : {} is not in options. plase use one of digit/upper/lower/english".format(class_type)); return None
    
    output_images = images[pick]
    output_labels = labels[pick] - margin
    
    return (output_images, output_labels)

def load_emnist(class_type = 'digit', add_channel = False):
    
    images_train, labels_train = pick_emnist_class('train', class_type, add_channel)
    images_test , labels_test  = pick_emnist_class('test', class_type, add_channel)
    
    return (images_train, labels_train),(images_test, labels_test)

def specific_load(dataset, add_channel = False, emnist_type = 'digit', choose = None):
    if dataset == 'mnist': data = load_mnist(add_channel = add_channel)
    elif dataset == 'emnist': data = load_emnist(class_type = emnist_type, add_channel = add_channel)
    elif dataset == 'fashion': data = load_fashion_mnist(add_channel = add_channel)
    elif dataset == 'cifar10': data = load_cifar10()
    else : return
    
    (images_train, labels_train),(images_test, labels_test) = data

    # Parse images and labels
    if choose != None:
        index = [i for i, v in enumerate(labels_train) if v in choose]
        images_train = images_train[index]
        labels_train = labels_train[index] - choose[0] # 라벨값 맞춰줌

        index = [i for i, v in enumerate(labels_test) if v in choose]
        images_test = images_test[index]
        labels_test = labels_test[index] - choose[0] # 라벨값 맞춰줌
    
    return (images_train, labels_train),(images_test, labels_test)