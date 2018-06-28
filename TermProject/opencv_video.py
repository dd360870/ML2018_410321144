import cv2
import dlib
import numpy as np
from sklearn.externals import joblib


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
"""You can download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"""

win = dlib.image_window()
svc = joblib.load("svc.pkl")
pca = joblib.load("pca.pkl")
label = joblib.load("label.pkl")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    win.clear_overlay()
    face_rects, scores, idx = detector.run(frame, 1)
    # dets = detector(frame, 1)

    if len(face_rects) > 0:
        faces = dlib.full_object_detections()
        for detection in face_rects:
            faces.append(predictor(frame, detection))
        images = dlib.get_face_chips(frame, faces, size=100, padding=0)
        data = np.array(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)).reshape(1, 10000)
        data_pca = pca.transform(data)
        result = svc.predict_proba(data_pca)
        indice = np.argsort(result).flatten()
        string = ""
        for i in range(np.alen(label)-2, np.alen(label)-7, -1):
            string += (label[indice[i]]+"({:.0f})".format(result[0][indice[i]]*100)+" ")
        cv2.putText(frame, string, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        # win.set_image(data.reshape(100, 100))
        # print(result)
        # dlib.hit_enter_to_continue()
        # print(result)
    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        text = "%2.2f(%d)" % (scores[i], idx[i])

        # 以方框標示偵測的人臉
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4, cv2.LINE_AA)

        # 標示分數
        cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    win.set_image(image)

cap.release()
cv2.destroyAllWindows()
