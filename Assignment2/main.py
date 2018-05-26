import numpy as np
import matplotlib.pyplot as plt
import struct
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

train_images_idx3_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\train-images.idx3-ubyte'
train_labels_idx1_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\train-labels.idx1-ubyte'
t10k_images_idx3_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\t10k-images.idx3-ubyte'
t10k_labels_idx1_ubyte = 'C:\\Users\\Ruzy\\Downloads\\train\\t10k-labels.idx1-ubyte'


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
        """for j in range(num_rows * num_cols):
            if images[i][j] > 250:
                images[i][j] = 1
            else:
                images[i][j] = 0"""
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


train_images = decode_idx3_ubyte(train_images_idx3_ubyte)
train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte)
test_images = decode_idx3_ubyte(t10k_images_idx3_ubyte)
test_labels = decode_idx1_ubyte(t10k_labels_idx1_ubyte)

pca = PCA(n_components=17, copy=False)
pca.fit(train_images)
train_data = pca.transform(train_images)
test_data = pca.transform(test_images)

"""for i in range(10):
    print(train_labels)
    plt.imshow(train_images[i].reshape((28, 28)), cmap='gray')
    plt.show()"""
K = 10

gmm = GaussianMixture(n_components=K)
gmm.fit(X=train_data)
p = 0
n = 0
indice_temp = np.zeros((K, 10))
test = gmm.predict(train_data)
for i in range(np.alen(test)):
    indice_temp[int(test[i])][int(train_labels[i])] += 1
indice = np.empty(K)
for i in range(K):
    max = 0
    lab = None
    for j in range(10):
        if indice_temp[i][j] > max:
            max = indice_temp[i][j]
            lab = j
    indice[i] = lab

result = gmm.predict(test_data)

for i in range(np.alen(result)):
    if indice[int(result[i])] == test_labels[i]:
        p += 1
    else:
        n += 1
print("%d / %d = %f" % (p, n, p / (p + n)))
