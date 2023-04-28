import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

train = glob.glob('train/*.png')
for image in train :
    print(image)
    img = cv2.imread(image,0)
    rows,cols = img.shape
    k = []
    for i in range(rows):
        for j in range(cols):
            k.append(img[i,j])
    image_vector = np.array(k)
    fourier_transform = np.fft.fft(image_vector)
    fft_power = np.abs(fourier_transform)
    phase = np.angle(fourier_transform)
    plt.plot(fft_power)
    plt.show()
    plt.plot(phase)
    plt.show()
