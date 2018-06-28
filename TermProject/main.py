from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import time
import cv2
import dlib
import glob
import os
import sys

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
"""You can download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"""
label = []

def read_files(path):
    X_data = []
    y_data = []
    for f in glob.glob(os.path.join(path, "*.jpg")):
        name = os.path.basename(f)
        if not (name[:name.find('_')] in label):
            label.append(name[:name.find('_')])
        img = dlib.load_rgb_image(f)
        dets = detector(img, 1)
        if len(dets) == 0:
            # cv2.imshow('mage', img)
            # cv2.waitKey(0)
            print("failed")
            continue
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(predictor(img, detection))
        images = dlib.get_face_chips(img, faces, size=100, padding=0)
        data = np.array(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)).reshape(10000)
        X_data.append(data)
        y_data.append(name[:name.find('_')])
    print("read_files({:s}) done.".format(path))
    return (X_data, y_data)


(X, y) = read_files("Face_Database/")
pca = PCA(n_components=150, whiten=True)
pca.fit(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

"""param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),param_grid)"""
t0 = time.time()
#clf.fit(X_train, y_train)
svc = SVC(probability=True, kernel="rbf")
svc.fit(X_train_pca, y_train)
t1 = time.time()
print("SVM fit done in {:.3f} sec".format(t1-t0))
pred = svc.predict(X_train_pca)
print("train acc : {:.2f}".format(accuracy_score(y_train, pred)*100.))
pred = svc.predict(X_test_pca)
print("test acc : {:.2f}".format(accuracy_score(y_test, pred)*100.))
# joblib.dump(svc, "svc.pkl")
# joblib.dump(pca, "pca.pkl")
# joblib.dump(np.array(label), "label.pkl")
# cv2.imshow("iamge", X_train[0].reshape((100, 100)))
# cv2.waitKey(0)