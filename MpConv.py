#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 18:48:35 2024

@author: york
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure
f1=cv2.imread('./test/1.png')
f2=cv2.imread('./test/2.png')
f3=cv2.imread('./test/3.jpeg')
f4=cv2.imread('./test/3.png')


img = f4
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
def apply_convolution(image, kernel_size, stride, padding):
    # 对图像应用卷积操作，这里使用高斯模糊模拟
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)




maxpooling_1=skimage.measure.block_reduce(img, 2, np.max)
conv_1 = apply_convolution(maxpooling_1, 3, 1, 0)
NORM_1=cv2.normalize(conv_1, None,0,255,cv2.NORM_MINMAX)


conv_2 = apply_convolution(img, 3, 1, 0)
NORM_2=cv2.normalize(conv_2, None,0,255,cv2.NORM_MINMAX)


conv_2_2 = apply_convolution(NORM_2, 3, 1, 0)
NORM_2_2=cv2.normalize(conv_2_2, None,0,255,cv2.NORM_MINMAX)

NORM_2_2 = cv2.resize(NORM_2_2,(NORM_1.shape[1],NORM_1.shape[0]))
cat = np.concatenate( (NORM_1, NORM_2_2),axis = 2)



# 可视化结果
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

plt.subplot(122)
plt.title('Simulated MpConv Result')
plt.imshow(cat[:, :, 1:5])

plt.axis('off')

plt.show()

