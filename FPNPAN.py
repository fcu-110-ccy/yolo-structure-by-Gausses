#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:33:43 2024

@author: york
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np

'''

img = cv2.imread('im1.png')

output1 = cv2.GaussianBlur(img, (5, 5), 0)
output2 = cv2.GaussianBlur(output1, (5, 5), 0)
output3 = cv2.GaussianBlur(output2, (5, 5), 0)
output4 = cv2.GaussianBlur(output3, (5, 5), 0)
img =output3-output4

rgb =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)                                    # 在圖表中繪製圖片
plt.show() 

import cv2'''


def norm(img):
    return img/np.max(img)


f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')




frame = f4
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame = cv2.resize(frame, (int(1280),int(720)))
frame = frame/np.max(frame)
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rows0, cols0, _channels = frame.shape
#output1 = cv2.GaussianBlur(frame, (5, 5), 0)
output1 = cv2.pyrDown(frame,dstsize=(int(cols0/2),  int(rows0/2))) 

rows1, cols1, _channels = output1.shape
#output2 = cv2.GaussianBlur(output1, (5, 5), 0)
output2 = cv2.pyrDown(output1,dstsize=(int(cols1/2),  int(rows1/2)))

rows2, cols2, _channels = output2.shape
#output3 = cv2.GaussianBlur(output2, (5, 5), 0
output3 = cv2.pyrDown(output2,dstsize=(int(cols2/2),  int(rows2/2)))

rows3, cols3, _channels = output3.shape
#output4 = cv2.GaussianBlur(output3, (5, 5), 0)
output4 = cv2.pyrDown(output3,dstsize=(int(cols3/2),  int(rows3/2)))
output444 = cv2.resize(output4, (int(1280/2),int(720/2)))
rows4, cols4, _channels = output4.shape
KKK=output4.copy()

#cv2.normalize(output4, output4, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output4, output4, 1.0, 0.0, cv2.NORM_MINMAX)


output5 = cv2.pyrUp(output4,dstsize=(int(cols4*2),  int(rows4*2)))

output6 = cv2.pyrUp(output5,dstsize=(int(cols3*2),  int(rows3*2)))
cv2.normalize(output2, output2, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output6, output6, 1.0, 0.0, cv2.NORM_L2)
output6 = output2-output6
#output6 = output5+output4
output7 = cv2.pyrUp(output6,dstsize=(int(cols2*2),  int(rows2*2)))
cv2.normalize(output1, output1, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output7, output7, 1.0, 0.0, cv2.NORM_L2)
output7 = output1-output7

#cv2.normalize(output7, output7, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output7, output7, 1.0, 0.0, cv2.NORM_MINMAX)


output8 = cv2.pyrDown(output7,dstsize=(int(cols1/2),  int(rows1/2)))
cv2.normalize(output6, output6, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output8, output8, 1.0, 0.0, cv2.NORM_L2)
output8 = output6-output8
output9 = cv2.pyrDown(output8,dstsize=(int(cols2/2),  int(rows2/2)))
cv2.normalize(output5, output5, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output9, output9, 1.0, 0.0, cv2.NORM_L2)
output9 = output5-output9
output10 = cv2.pyrDown(output9,dstsize=(int(cols3/2),  int(rows3/2)))
cv2.normalize(output4, output4, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output10, output10, 1.0, 0.0, cv2.NORM_L2)
output10 = output4-output10

#cv2.normalize(output10, output10, 1.0, 0.0, cv2.NORM_L2)
cv2.normalize(output8, output8, 1.0, 0.0, cv2.NORM_MINMAX)
cv2.normalize(output9, output9, 1.0, 0.0, cv2.NORM_MINMAX)
cv2.normalize(output10, output10, 1.0, 0.0, cv2.NORM_MINMAX)

img = output10


frame = cv2.resize(frame, (int(1280/2),int(720/2)))

output8 = cv2.resize(output8, (int(1280/2),int(720/2)))
output9 = cv2.resize(output9, (int(1280/2),int(720/2)))

output7 = cv2.resize(output7, (int(1280/2),int(720/2)))

output10 = cv2.resize(output10, (int(1280/2),int(720/2)))


#ret, notedge = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)     # 如果大於 127 就等於 255，反之等於 0。

#ret, edge = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)     # 如果大於 127 就等於 255，反之等於 0。

#EDGE = cv2.bitwise_and(notedge,edge,mask = notedge) 


#o1 = cv2.bitwise_and(img,img,mask = EDGE) 
#o2 = cv2.bitwise_and(img,img,mask = edge) 
  
#ret, notedge = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY_INV)     # 如果大於 127 就等於 255，反之等於 0。

#ret, edge = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)     # 如果大於 127 就等於 255，反之等於 0。

#EDGE = cv2.bitwise_and(notedge,edge,mask = notedge) 


#o1 = cv2.bitwise_and(img,img,mask = EDGE) 
#o2 = cv2.bitwise_and(img,img,mask = edge) 

# 可视化结果
plt.figure()
plt.subplot(331)
plt.title('Original Image')
plt.imshow(frame)
plt.axis('off')

plt.subplot(332)
plt.title('FPN 1st')
plt.imshow(KKK)

plt.axis('off')

plt.subplot(333)
plt.title('FPN 3rd')
plt.imshow(output7)

plt.axis('off')

plt.subplot(334)
plt.title('PAN 1st')
plt.imshow(output8)

plt.axis('off')

plt.subplot(335)
plt.title('PAN 2nd')
plt.imshow(output9)

plt.axis('off')

plt.subplot(336)
plt.title('PAN 3rd')
plt.imshow(output10)

plt.axis('off')

plt.show()





'''cv2.imshow('frame8', output8)
cv2.imshow('frame9', output9)
#cv2.imshow('frame', o1)
cv2.imshow('frame4', output444)
cv2.imshow('frame7', output7)
cv2.imshow('frame10', output10)
cv2.imshow('frame', frame)
cv2.waitKey(0)


# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()'''