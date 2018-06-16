import numpy as np
import struct
import time
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from pathlib import Path

train_images_idx3_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\train-images.idx3-ubyte'
train_labels_idx1_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\train-labels.idx1-ubyte'
t10k_images_idx3_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\t10k-images.idx3-ubyte'
t10k_labels_idx1_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\t10k-labels.idx1-ubyte'


def read_file():
    file = Path("train_data.npy")
    if not file.is_file():
        train_data = decode_idx3_ubyte(train_images_idx3_ubyte)
        np.save(file, train_data)
    else:
        train_data = np.load(file)
        print("file exist.")
    # ---------------------------------------------------------
    file = Path("train_label.npy")
    if not file.is_file():
        train_label = decode_idx1_ubyte(train_labels_idx1_ubyte)
        np.save(file, train_label)
    else:
        train_label = np.load(file)
        print("file exist.")
    # ---------------------------------------------------------
    file = Path("test_data.npy")
    if not file.is_file():
        test_data = decode_idx3_ubyte(t10k_images_idx3_ubyte)
        np.save(file, test_data)
    else:
        test_data = np.load(file)
        print("file exist.")
    # ---------------------------------------------------------
    file = Path("test_label.npy")
    if not file.is_file():
        test_label = decode_idx1_ubyte(t10k_labels_idx1_ubyte)
        np.save(file, test_label)
    else:
        test_label = np.load(file)
        print("file exist.")
    # ---------------------------------------------------------
    return train_data, train_label, test_data, test_label


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('%d, %d, %d*%d' % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows * num_cols))
    for i in range(num_images):
        if(i + 1) % 10000 == 0:
            print("processing : %d" % (i + 1))
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))#.reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('%d' % (i + 1))
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


(train_data, train_labels, test_data, test_labels) = read_file()


pca = PCA(n_components=35, copy=False)
pca.fit(train_data)


train_data = pca.transform(train_data)
test_data = pca.transform(test_data)
train_data = scale(train_data)
test_data = scale(test_data)
print("PCA done.")

knn = KNeighborsClassifier(n_neighbors=7)
t0 = time.time()
knn.fit(train_data, train_labels)
t1 = time.time()
print("kNN fit done {:.3f} sec.".format(t1-t0))
t0 = time.time()
re = knn.predict(train_data)
t1 = time.time()
print('kNN train accuracy: {:.2f}% in {:.3f} sec'.format(accuracy_score(train_labels, re)*100.0, t1-t0))
t0 = time.time()
re = knn.predict(test_data)
t1 = time.time()
print('kNN test accuracy: {:.2f}% in {:.3f} sec'.format(accuracy_score(test_labels, re)*100.0, t1-t0))

svc_model = svm.SVC()
t0 = time.time()
svc_model.fit(train_data, train_labels)
t1 = time.time()
print("SVM fit done {:.3f} sec.".format(t1-t0))
t0 = time.time()
re = svc_model.predict(train_data)
t1 = time.time()
print('SVM train accuracy: {:.2f}% in {:.3f} sec'.format(accuracy_score(train_labels, re)*100.0, t1-t0))
t0 = time.time()
re = svc_model.predict(test_data)
t1 = time.time()
print('SVM test accuracy: {:.2f}% in {:.3f} sec'.format(accuracy_score(test_labels, re)*100.0, t1-t0))
