#The effect of kernel size on feature selection

import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread("your image address")
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(img.shape)

filter3= np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
filter5= np.array([
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, 24, -1, -1],
    [-1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1]
])
filter7= np.array([
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 48, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1]
])

img_f3= cv2.filter2D(img, -1, filter3)
img_f5= cv2.filter2D(img, -1, filter5)
img_f7= cv2.filter2D(img, -1, filter7)

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.subplot(2, 2, 2)
plt.imshow(img_f3)
plt.title('filter 3')
plt.subplot(2, 2, 3)
plt.imshow(img_f5)
plt.title('filter 5')
plt.subplot(2, 2, 4)
plt.imshow(img_f7)
plt.title('filter 7')
plt.show()
