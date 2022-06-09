import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
from pyimagesearch.lenet import LeNet
import argparse
import random
import cv2
import os



EPOCHS = 25
INIT_LR = 1e-3
BS = 32

data = []
labels = []
dir_labels = ()
num_class = 0
DIRECTORY = r"D:/Woods\dataset"
print("[INFO] Finding Labels...")
CATEGORIES = ["Anh_dao", "Bach_dang_nk","Bach_xanh","Cam_lai","Cao_su","Dang_huong","Hoang_dan","Keo_lai","Lim","Mit_mat","Mun","Muong_den","Soi","Trai_li","Xoay"]
for file in os.listdir(DIRECTORY) :
	temp_tuple=(file,'null')
	dir_labels=dir_labels+temp_tuple
	dir_labels=dir_labels[:-1]
	print(dir_labels)
	num_class=num_class+1
	print(num_class)

print("[INFO] Loading Images...")
imagePaths = sorted(list(paths.list_images(DIRECTORY)))
random.seed(42)
random.shuffle(imagePaths)


for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (192, 256))
	image = img_to_array(image)
	data.append(image)


	label = imagePath.split(os.path.sep)[-2]
	
	for i in range(num_class) :
		if label == dir_labels[i] :
			label = i
	#print(label) 
	labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

print(data)
print(labels)
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)


trainY = to_categorical(trainY, num_classes=num_class)
testY = to_categorical(testY, num_classes=num_class)


aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] Compiling Model...")
model = LeNet.build(width=192, height=256, depth=3, classes=num_class)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print("[INFO] Training Network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1 )


print("[INFO] Saving Model...")
model.save("woods.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

print("[INFO] Completed...")