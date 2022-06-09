from distutils.log import error
from fileinput import filename
from re import T
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.uic import loadUi
from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import sys
import cv2
from WOODS import *
import numpy as np


class LoadQt(QMainWindow):
    filename = ""
    sizeimg=(641,461)
    model = load_model("woods.model")
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # create a timer
        self.timer = QTimer()
        # set timer timeout callback function
        self.timer.timeout.connect(self.viewCam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.controlTimer)
        self.ui.openimg.clicked.connect(self.open_img)
        self.ui.predict.clicked.connect(self.predict)


    
    
    # view camera
    def viewCam(self):
        # read image in BGR format
                 
            image = self.vs.read()
            # convert image to RGB format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
            # show image in img_label
            self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))
            #video(self.vs,self.ui.BSLB,self.ui.lb_bienso)
            
            
    

    # start/stop timer
    def controlTimer(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.vs = VideoStream(src=0).start()
            # start timer
            self.timer.start(20)
            # update control_bt text
            self.ui.control_bt.setText("Stop")
            # if timer is started
        else:
            # stop timer
            image = self.vs.read()
            self.timer.stop()
            # release video capture
            self.vs.stop()
            
            #save image
            showPic = cv2.imwrite("filename1.jpg",image)
            self.filename="filename1.jpg"
            self.showimage(self.logomain,self.sizeimg,self.ui.image_label)
            #self.showimage(self.logobs,self.sizebs,self.ui.lb_bienso)
            # update control_bt text
            self.ui.control_bt.setText("Start")

            
            
    
    def loadImage(self, fname):
        self.image = cv2.imread(fname)
        self.filename=fname
        self.tmp = self.image
        self.showimage(self.image,self.sizeimg,self.ui.image_label)

        
    
    
    #Hiển thị ảnh lên GUI
    def showimage(self,image,size,img_lb):
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize image
        resized = cv2.resize(image, size)
        # get image infos
        height, width, channel = resized.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(resized.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        img_lb.setPixmap(QPixmap.fromImage(qImg))
    #Mở hình ảnh
    def open_img(self):
        fname,_ = QFileDialog.getOpenFileName(self, 'Open File', '/Woods/test/', "Image Files (*)")
        #print(fname)
        if fname:
            self.loadImage(fname)
        else:
            print("Invalid Image")

    def predict(self):
        labels = ["Anh_dao", "Bach_dang_nk","Bach_xanh","Cam_lai","Cao_su","Dang_huong","Hoang_dan","Keo_lai","Lim","Mit_mat","Mun","Muong_den","Soi","Trai_li","Xoay"]
        
        img = cv2.imread(self.filename)
        img= cv2.resize(img,(192,256))
        img = img.astype("float32") / 255
        img = img_to_array(img)
        print(img.shape)
        img = np.expand_dims(img, 0)
        y_pred = self.model.predict(img)[0]
        rs= max(y_pred)
        print(rs)
        for i in range(15):
            if(y_pred[i]== rs):
                label= labels[i]

        self.ui.RSLB.setText(label)
        label2 = "{} : Tỉ lệ: {:.2f}%".format(label, rs * 100)


        print("Kết quả dự đoán là loại gỗ :",label2)
        print(y_pred)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = LoadQt()
    mainWindow.show()

    sys.exit(app.exec_())