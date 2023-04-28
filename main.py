import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from Auto_Gradient_Boosting import AGB
from Artificial_Neural_Network_Classifier import artificialneuralnetwork_classifier

fruitimg = glob.glob('train/*.png')
x1_matrix = []
x2_matrix = []
y_matrix = []
y_not = []
for image in fruitimg :
    print(image)
    img = cv2.imread(image,0)
    rows,cols = img.shape
    k = []
    for i in range(rows):
        for j in range(cols):
            k.append(img[i,j])

    image_vector = np.array(k)
    fourier_transform = np.fft.fft(image_vector)
    fft_power = np.abs(fourier_transform[1:101]).astype(int)
    phase = np.angle(fourier_transform[1:101]).astype(int)
    fft_list = phase.tolist()
    x1_matrix.append(fft_list)
    fft_list = fft_power.tolist()
    x2_matrix.append(fft_list)
    if "apple" in image.lower():
      y_matrix.append([1])
      y_not.append([0])
    else :
      y_matrix.append([0])
      y_not.append([1])



x1=np.matrix(x1_matrix)
x2=np.matrix(x2_matrix)
y=np.matrix(y_matrix)
y2=np.matrix(y_not)

learning_rate = 0.009999
agb1 = AGB(x1,y,learning_rate)
agb2 = AGB(x2,y,0.0000009999)
X_F = []
fruitimg = glob.glob('train/*.png')
for image in fruitimg :
  img = cv2.imread(image,0)
  rows,cols = img.shape
  k = []
  for i in range(rows):
      for j in range(cols):
          k.append(img[i,j])

  image_vector = np.array(k)
  fourier_transform = np.fft.fft(image_vector)
  fft_power = np.abs(fourier_transform[1:101]).astype(int)
  phase = np.angle(fourier_transform[1:101]).astype(int)
  fft_list = phase.tolist()
  fft_list2 = fft_power.tolist()
  X1 = np.matrix([fft_list])
  X2 = np.matrix([fft_list2])
  p1 = agb1.predict(X1)
  p2 = agb2.predict(X2)
  print(image,p1,p2)
  X_F.append([p1,p2])

x_f = np.matrix(X_F)


ANN = AGB(x_f,y,0.09999)
def predict(image):
    img = cv2.imread(image,0)
    rows,cols = img.shape
    k = []
    for i in range(rows):
        for j in range(cols):
            k.append(img[i,j])

    image_vector = np.array(k)
    fourier_transform = np.fft.fft(image_vector)
    fft_power = np.abs(fourier_transform[1:101])
    phase = np.angle(fourier_transform[1:101])
    fft_list = phase.tolist()
    fft_list2 = fft_power.tolist()
    X1 = np.matrix([fft_list])
    X2 = np.matrix([fft_list2])
    p1 = agb1.predict(X1)
    p2 = agb2.predict(X2)
    X_F=np.matrix([[p1,p2]])
    print(image,ANN.predict(X_F))

fruitimg = glob.glob('test/*.png')
for image in fruitimg :
    predict(image)
