from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import time
import cv2
import dlib
import glob
import os

class TermProject:
    def __init__(self):
        self.detector = detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        """You can download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"""
        self.pca = PCA(n_components=150, whiten=True)
        self.svc = SVC(probability=True, kernel="rbf")
        self.label = None
        self.X_data = None
        self.y_data = None
    def read_files(self, path):
        label = []
        X_data = []
        y_data = []
        for f in glob.glob(os.path.join(path, "*.jpg")):
            name = os.path.basename(f)
            if not (name[:name.find('_')] in label):
                label.append(name[:name.find('_')])
            img = dlib.load_rgb_image(f)
            dets = self.detector(img, 1)
            if len(dets) == 0: # No faces found in the pic
                print("detection failed ({:s})".format(f))
                continue
            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(self.predictor(img, detection))
            images = dlib.get_face_chips(img, faces, size=100, padding=0)
            data = np.array(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)).reshape(10000)
            X_data.append(data)
            y_data.append(name[:name.find('_')])
        print("read_files({:s}) done.".format(path))
        self.X_data = X_data
        self.y_data = y_data
        self.label = label
    def run(self):
        self.pca.fit(self.X_data)
        X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.1)
        X_train_pca =self.pca.transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        t0 = time.time()
        self.svc.fit(X_train_pca, y_train)
        t1 = time.time()
        print("SVM fit done in {:.3f} sec".format(t1 - t0))
        pred = self.svc.predict(X_train_pca)
        print("train acc : {:.2f}%".format(accuracy_score(y_train, pred) * 100.))
        pred = self.svc.predict(X_test_pca)
        print("test acc : {:.2f}%".format(accuracy_score(y_test, pred) * 100.))
        print("--------------------")
    def save_mode(self):
        joblib.dump(svc, "svc.pkl")
        joblib.dump(pca, "pca.pkl")
        joblib.dump(np.array(label), "label.pkl")


test = TermProject()
test.read_files("Face_Database/")
test.run()
test.run()
test.run()