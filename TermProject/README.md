# Term Project
Face Recognition
### 說明
###### 預處理：
1.先使用dlib將每張圖片的臉部抓出來，轉正成100*100的圖片，轉置為10000的向量

    img = dlib.load_rgb_image(f)
    dets = self.detector(img, 1) #detect faces
    if len(dets) == 0: #No faces found in this pic
      print("detection failed ({:s})".format(f))
      continue
    faces = dlib.full_object_detections()
    for detection in dets:
      faces.append(self.predictor(img, detection))
    images = dlib.get_face_chips(img, faces, size=100, padding=0) #get face image(100*100)
    data = np.array(cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)).reshape(10000)

2.使用sklearn.pca將10000的向量處理成150，並隨機分資料的10%為test_data，90%為train_data

    self.pca = PCA(n_components=150, whiten=True)
    self.pca.fit(self.X_data)
    X_train, X_test, y_train, y_test = train_test_split(self.X_data, self.y_data, test_size=0.1)
    X_train_pca =self.pca.transform(X_train)
    X_test_pca = self.pca.transform(X_test)

###### 訓練：
使用sklearn.svm / SVC() 訓練train_data
### 桌面環境
Windows 10 x64<br>
Python x64 3.6.5<br>
scikit-learn           0.19.1<br>
dlib                   19.13.1<br>
### 結果討論

    ↓執行結果↓
    --------------------
    SVM fit done in 0.782 sec
    train acc : 100.00%
    test acc : 81.54%
    --------------------
    SVM fit done in 0.757 sec
    train acc : 100.00%
    test acc : 83.08%
    --------------------
    SVM fit done in 0.762 sec
    train acc : 100.00%
    test acc : 80.00%
    --------------------
因每次train_data & test_data抓出來的部分隨機選擇，所以每次辨識的結果都不太一樣，test acc大約在70~85%左右
### 參考資料
<a href="http://dlib.net/">dlib C++ Library</a>
