
import cv2
import matplotlib.pyplot as plt
import numpy as np


f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')

frame =f3



output1 = cv2.GaussianBlur(frame, (5, 5), 0)
   



img = cv2.normalize(output1, None,0,255,cv2.NORM_MINMAX)


frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.title('original')
plt.imshow(frame)

plt.subplot(122)
plt.title('BN')
plt.imshow(img)

plt.show()