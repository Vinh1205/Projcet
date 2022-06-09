from logging import basicConfig
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import imutils
import cv2
labels = ["Anh_dao", "Bach_dang_nk","Bach_xanh","Cam_lai","Cao_su","Dang_huong","Hoang_dan","Keo_lai","Lim","Mit_mat","Mun","Muong_den","Soi","Trai_li","Xoay"]
model = load_model("woods.model")
img = cv2.imread("test/cs2.jpg")
img= cv2.resize(img,(192,256))
img = img.astype("float32") / 255
img = img_to_array(img)
print(img.shape)
img = np.expand_dims(img, 0)
y_pred = model.predict(img)[0]
rs= max(y_pred)
print(rs)
for i in range(15):
    if(y_pred[i]== rs):
        label= labels[i]
label = "{} : Tỉ lệ: {:.2f}%".format(label, rs * 100)

print("Kết quả dự đoán là loại gỗ :",label)
print(y_pred)
