from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import time
import cv2

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    """
      name                              | miss 
    ------------------------------------+------
    haarcascade_frontalcatface          | 
    haarcascade_frontalcatface_extended | 
    haarcascade_frontalface_alt         | 
    haarcascade_frontalface_alt_tree    | 
    haarcascade_frontalface_alt2        | 
    haarcascade_frontalface_default     | 0.08
    lbpcascade_frontalface              | 
    """
    face_cascade = cv2.CascadeClassifier("opencv_files/haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    if len(faces) == 0:
        #print("No face")
        return None
    (x, y, w, h) = faces[0]
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    crop_img = gray[y:y+h, x:x+w]
    crop_img = cv2.resize(crop_img, (200, 200))
    return crop_img

def read_files(path):
    X_data = []
    y_data = []
    miss = 0
    filenames = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(np.alen(filenames)):
        print('.', end='', flush=True)
        if i % 50 == 0:
            print('\r', end='', flush=True)
        im = cv2.imread(path+filenames[i])
        im = cv2.resize(im, (240, 320))
        result = detect_face(im)
        if result is not None:
            X_data.append(np.array(result).flatten())
            y_data.append(filenames[i][:filenames[i].find("_")])
            #y_train.append(int(filenames[i][1:3]))
        else:
            miss += 1
    print("read_files() done. miss rate = {:.2f}".format(miss/np.alen(filenames)))
    return (X_data, y_data)


(X_train, y_train) = read_files("Face_Database/")
(X_test, y_test) = read_files("Face_Database/test/")
pca = PCA(n_components=70, copy=False, whiten=False)
pca.fit(X_train)
# X_train = pca.transform(X_train)
X_train = scale(X_train)
# X_test = pca.transform(X_test)
X_test = scale(X_test)
"""param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),param_grid)"""
t0 = time.time()
#clf.fit(X_train, y_train)
svc = SVC()
svc.fit(X_train, y_train)
t1 = time.time()
print("SVM fit done in {:.3f} sec".format(t1-t0))
pred = svc.predict(X_train)
print("train acc : {:.2f}".format(accuracy_score(y_train, pred)*100.))
pred = svc.predict(X_test)
print("test acc : {:.2f}".format(accuracy_score(y_test, pred)*100.))
