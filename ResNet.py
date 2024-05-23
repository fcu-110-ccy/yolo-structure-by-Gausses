

import cv2
import matplotlib.pyplot as plt
import numpy as np


f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')

frame =f4


output1 = cv2.GaussianBlur(frame, (1, 1), 0)
   
output2 = cv2.GaussianBlur(output1, (3, 3), 0)
   

img = frame+output2


img = cv2.normalize(img, None,0,255,cv2.NORM_MINMAX)


frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.title('original')
plt.imshow(frame)

plt.subplot(122)
plt.title('residual')
plt.imshow(img)

plt.show()